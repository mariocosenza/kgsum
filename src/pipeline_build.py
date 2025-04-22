from __future__ import annotations

import ast
import logging
import os
import pickle
import re
import warnings
from collections import Counter
from enum import Enum, auto
from typing import Any, Tuple, NewType, TypeAlias

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging, TrainerCallback, TrainerState,
    TrainerControl
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

hf_logging.set_verbosity_error()
np.seterr(invalid='ignore')
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'token_pattern' will not be used*")
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


FeatureLabels: TypeAlias = str | list[str]
TextData = NewType('TextData', str)

class ClassifierType(Enum):
    SVM = auto()
    NAIVE_BAYES = auto()
    KNN = auto()
    MISTRAL = auto()

def is_uri(token):
    uri_regex = re.compile(
        r"^(?:https?|ftp|file)://[^\s<>'\"`]+$|^www\.[^\s<>'\"`]+$|^[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-zA-Z]{2,}.*$"
    )
    return bool(uri_regex.match(token))

def hybrid_tokenizer(text):
    tokens = text.split()
    processed = []
    for tok in tokens:
        if is_uri(tok):
            processed.append(tok)
        else:
            processed.extend(re.findall(r"(?u)\b\w\w+\b", tok))
    return processed

def get_custom_tfidf_vectorizer(**kwargs):
    return TfidfVectorizer(
        ngram_range=(1, 1),
        lowercase=True,
        tokenizer=hybrid_tokenizer,
        token_pattern=None,
        norm='l2',
        **kwargs
    )

def get_custom_count_vectorizer(**kwargs):
    return CountVectorizer(
        tokenizer=hybrid_tokenizer,
        token_pattern=None,
        **kwargs
    )

def majority_vote(predictions: list[Any]) -> Any:
    if isinstance(predictions, str):
        return predictions
    filtered_predictions = [p for p in predictions if p is not None]
    if not filtered_predictions:
        logger.info("No valid predictions available for majority vote.")
        return None
    if isinstance(filtered_predictions[0], (tuple, list)) and len(filtered_predictions[0]) > 1:
        candidate_info = {}
        for tup in filtered_predictions:
            label = tup[0]
            try:
                weight = float(tup[1])
            except Exception:
                weight = 1.0
            candidate_info[label] = candidate_info.get(label, 0.0) + weight
        if not candidate_info:
            logger.info("No weighted predictions available for majority vote.")
            return None
        best_label = max(candidate_info.items(), key=lambda x: x[1])[0]
        logger.info("Majority vote: candidate '%s' total weight: %.4f", best_label, candidate_info[best_label])
        return best_label
    best_label = Counter(filtered_predictions).most_common(1)[0][0]
    logger.info("Majority vote result: candidate '%s'", best_label)
    return best_label

def _predict_category_for_instance(
    models: dict[str, 'KnowledgeGraphClassifier'],
    instance: dict[str, Any]
) -> Any | None:
    votes: list[tuple[Any, float, str]] = []
    for feature, model in models.items():
        value = instance.get(feature)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
            feature_data = [value]
        elif isinstance(value, (list, set, tuple)):
            feature_data = [str(item).strip() for item in value if item and str(item).strip()]
            if not feature_data:
                continue
        else:
            value = str(value).strip()
            if not value:
                continue
            feature_data = [value]
        try:
            pred = model.predict(feature_data)[0]
            score = getattr(model, "accuracy", 0.5)
            votes.append((pred, score, feature))
        except Exception as err:
            logger.error("Prediction error for %s: %s", feature, err)
    label_votes = [v[0] for v in votes]
    return majority_vote(label_votes)

def predict_category_multi(
    models: dict[str, 'KnowledgeGraphClassifier'],
    instance: dict[str, Any] | pd.DataFrame
) -> Any | list[Any | None]:
    if isinstance(instance, pd.DataFrame):
        return instance.apply(lambda row: _predict_category_for_instance(models, row.to_dict()), axis=1).tolist()
    return _predict_category_for_instance(models, instance)

def remove_empty_rows(frame: pd.DataFrame, labels: str | list[str]) -> pd.DataFrame:
    if isinstance(labels, str):
        labels = [labels]
    result = frame.copy()
    for label in labels:
        result = result.dropna(subset=[label])
        result = result[result[label] != '']
    return result

def oversample_dataframe(df: pd.DataFrame, target_label: str, max_factor: float = 1.2) -> pd.DataFrame:
    counts = df[target_label].value_counts()
    logger.info("Original class counts: %s", counts.to_dict())
    max_count = counts.max()
    groups = []
    for label, group in df.groupby(target_label):
        target_count = min(max_count, int(len(group) * max_factor))
        if len(group) < target_count:
            groups.append(group.sample(target_count, replace=True, random_state=42))
        else:
            groups.append(group)
    if not groups:
        return pd.DataFrame(columns=df.columns)
    oversampled = pd.concat(groups).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("After oversampling: %s", oversampled[target_label].value_counts().to_dict())
    return oversampled

def train_multiple_models(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    target_label: str = 'category',
    classifier_type: ClassifierType = ClassifierType.NAIVE_BAYES,
) -> Tuple[dict[str, 'KnowledgeGraphClassifier'], dict[str, Any]]:
    models: dict[str, KnowledgeGraphClassifier] = {}
    training_results: dict[str, Any] = {}
    for feature in tqdm(feature_columns, desc="Training models"):
        df_feature = remove_empty_rows(training_data, [feature, target_label])
        if df_feature.empty:
            logger.info("Skipping '%s': no data available.", feature)
            continue
        if df_feature[target_label].nunique() < 2:
            logger.info("Skipping '%s': only one unique class: %s", feature, df_feature[target_label].unique())
            continue
        logger.info("Training model for '%s' with %s examples.", feature, len(df_feature))
        model = KnowledgeGraphClassifier(classifier_type=classifier_type, balance_classes=True)
        try:
            result = model.train(df_feature, feature, target_label=target_label)
        except Exception as err:
            logger.error("Error training model for '%s': %s", feature, err)
            continue
        models[feature] = model
        training_results[feature] = result
    return models, training_results


class TqdmDataCallback(TrainerCallback):
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.pbar = None
        self.current_step = 0

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        from tqdm import tqdm
        if state.is_local_process_zero:
            self.pbar = tqdm(total=self.total_steps, desc="Training samples", position=0, leave=True)
            self.current_step = 0

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_local_process_zero and self.pbar is not None:
            step = state.global_step - self.current_step
            if step > 0:
                self.pbar.update(step)
            self.current_step = state.global_step

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_local_process_zero and self.pbar is not None:
            self.pbar.n = self.total_steps
            self.pbar.refresh()
            self.pbar.close()
            self.pbar = None

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class KnowledgeGraphClassifier:
    def __init__(self, classifier_type: ClassifierType =ClassifierType.NAIVE_BAYES, balance_classes=False, vectorizer_type="tfidf", feature_labels=None, **kwargs):
        self.classifier_type: ClassifierType= classifier_type
        self.balance_classes = balance_classes
        self.vectorizer_type = vectorizer_type
        self.feature_labels = feature_labels
        self.vectorizer = self._init_vectorizer(**kwargs)
        self.model: Any = None
        self.mistral_tokenizer: Any = None
        self.target_label_mapping: dict[Any, int] | None = None
        self.target_label_inverse_mapping: dict[int, Any] | None = None
        self.feature_labels: FeatureLabels | None = None
        self.max_length: int | None = None
        self.accuracy: float = 0.0

    def _init_vectorizer(self, **kwargs):
        if self.vectorizer_type == "tfidf":
            return get_custom_tfidf_vectorizer(**kwargs)
        else:
            return get_custom_count_vectorizer(**kwargs)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\[\s*\]', '', text)
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\{\s*\}', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _prepare_features(self, frame, feature_labels):
        # Fast path: no lists, no parsing
        if isinstance(feature_labels, (list, tuple)):
            # Join all columns as strings, vectorized
            return frame.loc[:, feature_labels].astype(str).agg(" ".join, axis=1)
        else:
            col = frame[feature_labels]

            # Only parse rows that look like a list, otherwise just str
            def process(val):
                if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
                    try:
                        parsed = ast.literal_eval(val)
                        if isinstance(parsed, (list, tuple, set)):
                            return " ".join(map(str, parsed))
                    except Exception:
                        pass
                return str(val)

            return col.map(process)


    def _get_param_distributions(self, classic_type: ClassifierType = None, y_train=None) -> dict:
        ctype = classic_type or self.classifier_type
        safe_max = None
        if y_train is not None and len(y_train) > 0:
            unique_classes = len(np.unique(y_train))
            if unique_classes > 1:
                safe_max = max(2, min(15, int(0.8 * len(y_train) / unique_classes)))

        if ctype == ClassifierType.SVM:
            return {
                "classifier__C": np.logspace(-3, 2, 20),
                "classifier__kernel": ["linear", "rbf"],
                "classifier__class_weight": ["balanced"],
                "classifier__gamma": ["scale", "auto"] + list(np.logspace(-3, 1, 10))
            }
        elif ctype == ClassifierType.NAIVE_BAYES:
            return {
                "classifier__alpha": np.linspace(0.01, 10, 50),
                "classifier__fit_prior": [True, False]
            }
        elif ctype == ClassifierType.KNN:
            n_neighbors_range = list(range(1, safe_max + 1)) if safe_max else list(range(1, 16))
            return {
                "classifier__n_neighbors": n_neighbors_range,
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan", "chebyshev", "cosine"],
                "classifier__leaf_size": list(range(10, 51, 5))
            }
        return {}

    def _get_base_estimators(self):
        svm_clf = svm.SVC(probability=True, kernel="rbf", C=10, class_weight="balanced", random_state=42)
        nb_clf = MultinomialNB(alpha=0.5)
        lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        rf_clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        knn_clf = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
        return [
            ("svm", svm_clf),
            ("nb", nb_clf),
            ("lr", lr_clf),
            ("rf", rf_clf),
            ("knn", knn_clf)
        ]

    def train(
            self,
            frame: pd.DataFrame,
            feature_labels: FeatureLabels,
            target_label: str = 'category',
            max_length: int = 256,
    ) -> dict[str, Any]:
        import random
        from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

        clf_type = self.classifier_type

        if clf_type == ClassifierType.MISTRAL:
            return self.train_mistral(frame, feature_labels, target_label=target_label, max_length=max_length)


        self.feature_labels = feature_labels
        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]
        unique_labels = sorted(data_y.unique())
        if len(unique_labels) < 2:
            raise ValueError(f"Need at least 2 unique classes in target '{target_label}', found {len(unique_labels)}")

        self.target_label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.target_label_inverse_mapping = {idx: label for label, idx in self.target_label_mapping.items()}

        y_numeric = data_y.map(self.target_label_mapping).values

        if len(data_x) != len(y_numeric):
            raise ValueError("Feature and target lengths mismatch after preparation.")
        if len(data_x) == 0:
            raise ValueError("No data available for training after preparation.")

        X_train, X_test, y_train, y_test = train_test_split(
            data_x,
            y_numeric,
            test_size=0.20,
            random_state=42,
            stratify=y_numeric
        )

        if self.balance_classes:
            value_counts = pd.Series(y_train).value_counts()
            if len(value_counts) > 1 and (value_counts.min() / value_counts.max()) < 0.75:
                logger.info("Unbalanced classes detected in train; applying oversampling.")
                df_train = pd.DataFrame({'X': X_train, 'y': y_train})
                df_train = oversample_dataframe(df_train, 'y', max_factor=2.0)
                if not df_train.empty:
                    X_train = df_train['X']
                    y_train = df_train['y'].values
                    logger.info("After oversampling, train class counts: %s",
                                pd.Series(y_train).value_counts().to_dict())
                else:
                    logger.warning("Oversampling resulted in an empty dataframe. Proceeding without oversampling.")

        clf_type = self.classifier_type
        if clf_type == ClassifierType.SVM:
            estimator = svm.SVC(probability=True, random_state=42)
            param_distributions = {
                "classifier__C": np.logspace(-2, 2, 8),
                "classifier__kernel": ["linear", "rbf", "sigmoid"],
                "classifier__gamma": ["scale", "auto"],
            }
        elif clf_type == ClassifierType.NAIVE_BAYES:
            estimator = MultinomialNB()
            param_distributions = {
                "classifier__alpha": np.linspace(0.1, 2, 7),
                "classifier__fit_prior": [True, False],
            }
        elif clf_type == ClassifierType.KNN:
            estimator = KNeighborsClassifier()
            param_distributions = {
                "classifier__n_neighbors": list(range(2, 7)),
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            }
        else:
            raise ValueError("Unsupported classic model type.")

        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", estimator),
        ])

        if param_distributions:
            single_param = random.choice(list(param_distributions.keys()))
            single_value = param_distributions[single_param]
            print(f"[LOG] Random search parameter used: {single_param} = {single_value}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        n_iter = 10

        def get_total_possible(param_distributions):
            total = 1
            for v in param_distributions.values():
                total *= len(v)
            return total

        total_possible = get_total_possible(param_distributions)
        n_fits = min(n_iter, total_possible)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='f1_weighted',
            verbose=3,
            n_jobs=1,
            error_score=0.0,
            random_state=42,
            return_train_score=False
        )

        print(f"Starting RandomizedSearchCV with up to {n_fits} parameter sets...")
        search.fit(X_train, y_train)

        print("[LOG] All fit parameter sets tried during RandomizedSearchCV:")
        for i, params in enumerate(search.cv_results_['params']):
            print(f"Set {i + 1}: {params}")

        best_estimator = search.best_estimator_
        y_pred_numeric = best_estimator.predict(X_test)
        final_f1 = f1_score(y_test, y_pred_numeric, average='weighted', zero_division=0)
        final_acc = accuracy_score(y_test, y_pred_numeric)
        self.accuracy = final_f1
        conf_mat = confusion_matrix(y_test, y_pred_numeric)
        self.model = best_estimator
        report = classification_report(
            y_test, y_pred_numeric,
            target_names=[str(self.target_label_inverse_mapping[i]) for i in
                          range(len(self.target_label_inverse_mapping))],
            zero_division=0
        )
        logger.info("\n" + report)
        logger.info("Best Params: %s", search.best_params_)
        logger.info("Final Model F1: %.4f, Accuracy: %.4f", final_f1, final_acc)
        print("\nConfusion Matrix:\n", conf_mat)
        print("\n=== Misclassified Test Examples ===")
        for text, true_idx, pred_idx in zip(X_test, y_test, y_pred_numeric):
            input_tokens = self.vectorizer.build_tokenizer()(text)
            if true_idx != pred_idx:
                print(f"INPUT TOKENS: {input_tokens}")
                print(
                    f"TRUE LABEL: {self.target_label_inverse_mapping[true_idx]} | PREDICTED: {self.target_label_inverse_mapping[pred_idx]}\n")
        return {
            'best_params': search.best_params_,
            'final_f1': final_f1,
            'final_accuracy': final_acc,
            'confusion_matrix': conf_mat.tolist(),
            'feature_labels': feature_labels,
            'target_label_mapping': self.target_label_mapping,
            'classifier_type': self.classifier_type,
            'classification_report': report
        }

    def train_mistral(
            self,
            frame: pd.DataFrame,
            feature_labels: FeatureLabels,
            target_label: str = 'category',
            max_length: int = 256,
            qlora_r: int = 16,
            qlora_alpha: int = 32,
            batch_size: int = 8,
            epochs: int = 10
    ) -> dict[str | Any, float | int | dict[str, int] | str | list[str] | Any] | None:
        import os
        import gc
        import torch
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig
        )
        from sklearn.metrics import accuracy_score, f1_score

        torch.backends.cudnn.benchmark = True

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Mistral training.")
        if LoraConfig is None or get_peft_model is None or prepare_model_for_kbit_training is None:
            raise ImportError("peft is not installed. Please install with pip install peft")

        frame = frame.reset_index(drop=True)
        frame = frame.dropna(subset=[target_label])
        frame = frame[frame[target_label] != '']

        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]

        unique_labels = sorted(data_y.unique())
        if len(unique_labels) < 2:
            raise ValueError(f"Need at least 2 unique classes in target '{target_label}', found {len(unique_labels)}")
        self.target_label_mapping = {str(label): int(idx) for idx, label in enumerate(unique_labels)}
        self.target_label_inverse_mapping = {int(idx): str(label) for idx, label in enumerate(unique_labels)}
        num_classes = len(unique_labels)

        texts = [self._clean_text(str(text)) for text in data_x.tolist()]
        y_numeric = [self.target_label_mapping[str(label)] for label in data_y.tolist()]

        X_train, X_temp, y_train, y_temp = train_test_split(
            texts,
            y_numeric,
            test_size=0.20,
            random_state=42,
            stratify=y_numeric
        )
        X_val, X_test, y_val, y_test = [], [], [], []
        if len(X_temp) > 1 and len(set(y_temp)) > 1:
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
            )
        elif len(X_temp) > 0:
            X_val, y_val = X_temp, y_temp

        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self.mistral_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.mistral_tokenizer.padding_side = "right"
        if self.mistral_tokenizer.pad_token is None:
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token
            self.mistral_tokenizer.pad_token_id = self.mistral_tokenizer.eos_token_id
        self.max_length = max_length

        enc_train = self.mistral_tokenizer(X_train, truncation=True, padding='max_length', max_length=self.max_length)
        train_dataset = HFDataset(enc_train, y_train)
        enc_val = self.mistral_tokenizer(X_val, truncation=True, padding='max_length', max_length=self.max_length) if X_val else None
        val_dataset = HFDataset(enc_val, y_val) if X_val else None
        enc_test = self.mistral_tokenizer(X_test, truncation=True, padding='max_length', max_length=self.max_length) if X_test else None
        test_dataset = HFDataset(enc_test, y_test) if X_test else None

        data_collator = DataCollatorWithPadding(
            tokenizer=self.mistral_tokenizer,
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=8
        )

        output_dir = "../data/trained/mistral-results"
        logging_dir = "../data/trained/mistral-logs"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_classes,
            quantization_config=quantization_config,
            device_map="auto",
            use_cache=False,
            low_cpu_mem_usage=True,
            pad_token_id=self.mistral_tokenizer.pad_token_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model.config.pad_token_id = self.mistral_tokenizer.pad_token_id
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=qlora_r,
            lora_alpha=qlora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=batch_size,
            logging_steps=100,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            logging_dir=logging_dir,
            bf16=True,
            fp16=False,
            report_to="none",
            seed=42,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            greater_is_better=True,
            save_total_limit=2,
            warmup_steps=2,
            optim="paged_adamw_8bit",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average='weighted', zero_division=0)
            }

        total_train_examples = len(train_dataset)
        callbacks = [TqdmDataCallback(total_steps=total_train_examples)]

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        try:
            trainer.train()
        except Exception as e:
            raise
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        val_results = trainer.evaluate(eval_dataset=val_dataset) if val_dataset is not None else {}
        test_results = trainer.evaluate(eval_dataset=test_dataset) if test_dataset is not None else {}

        val_f1 = float(val_results.get("eval_f1", 0.0))
        val_acc = float(val_results.get("eval_accuracy", 0.0))
        test_f1 = float(test_results.get("eval_f1", 0.0)) if test_dataset is not None else 0.0
        test_acc = float(test_results.get("eval_accuracy", 0.0)) if test_dataset is not None else 0.0

        self.accuracy = val_f1
        self.model = trainer.model

        return {
            "val_f1": val_f1,
            "val_accuracy": val_acc,
            "test_f1": test_f1,
            "test_accuracy": test_acc,
            "num_classes": int(num_classes),
            "max_length": int(self.max_length),
            "target_label_mapping": dict(self.target_label_mapping),
            "feature_labels": feature_labels,
            "classifier_type": str(self.classifier_type),
            "model_output_dir": str(training_args.output_dir)
        }

    def predict(
            self,
            data: str | list[str] | pd.Series | pd.DataFrame,
            feature_labels: FeatureLabels | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        if feature_labels is None and self.feature_labels:
            feature_labels = self.feature_labels
        elif feature_labels is None and self.feature_labels is None:
            raise ValueError("feature_labels must be provided for prediction if not set during training.")

        if isinstance(data, pd.DataFrame):
            texts = self._prepare_features(data, feature_labels).tolist()
        elif isinstance(data, pd.Series):
            if isinstance(feature_labels, list) and len(feature_labels) > 1:
                raise ValueError("Cannot process pd.Series with multiple feature_labels. Provide a DataFrame.")
            temp_df = data.to_frame(name=feature_labels if isinstance(feature_labels, str) else feature_labels[0])
            texts = self._prepare_features(temp_df, feature_labels).tolist()
        elif isinstance(data, list):
            texts = [self._clean_text(str(item)) for item in data]
        elif isinstance(data, str):
            texts = [self._clean_text(data)]
        else:
            raise ValueError("Unsupported data type for prediction. Use str, list, pd.Series, or pd.DataFrame.")

        if not texts:
            return np.array([])

        if self.classifier_type == ClassifierType.MISTRAL:
            import torch
            from tqdm import tqdm
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for Mistral inference.")
            if self.mistral_tokenizer is None:
                raise ValueError("Mistral tokenizer not available. Load the model first.")
            if self.target_label_inverse_mapping is None:
                raise ValueError("Target label mapping not available. Load the model first.")

            results = []
            batch_size = 16
            self.model.eval()
            for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", leave=False):
                batch = texts[i:i + batch_size]
                inputs = self.mistral_tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                results.extend(preds)

            if self.target_label_inverse_mapping:
                return np.array([self.target_label_inverse_mapping.get(int(idx), None) for idx in results])
            return np.array(results)

        else:
            if not hasattr(self.model, 'predict'):
                raise ValueError("Loaded classic model does not have a predict method.")
            try:
                numerical_preds = self.model.predict(texts)
            except Exception as e:
                logger.error("Classic model prediction error: %s", e)
                raise ValueError(f"Prediction failed for classic model: {e}")

            if self.target_label_inverse_mapping:
                return np.array([self.target_label_inverse_mapping.get(int(idx), None) for idx in numerical_preds])
            return numerical_preds

    def save(self, filepath: str | None = None) -> None:
        if self.model is None:
            raise ValueError("No trained model to save.")
        if filepath is None:
            os.makedirs('../data/trained', exist_ok=True)
            filepath = f'../data/trained/model-{self.classifier_type}.pkl'
        data_to_save: dict[str, Any] = {
            'classifier_type': self.classifier_type,
            'target_label_mapping': self.target_label_mapping,
            'target_label_inverse_mapping': self.target_label_inverse_mapping,
            'feature_labels': self.feature_labels,
            'max_length': self.max_length,
            'accuracy': self.accuracy
        }
        model_dir = None
        try:
            if self.classifier_type == ClassifierType.MISTRAL:
                model_dir = filepath.replace('.pkl', '')
                os.makedirs(model_dir, exist_ok=True)
                self.model.save_pretrained(model_dir)
                if self.mistral_tokenizer:
                    self.mistral_tokenizer.save_pretrained(model_dir)
                data_to_save['model_dir'] = model_dir
                metadata_path = filepath
            else:
                data_to_save['vectorizer'] = self.vectorizer
                data_to_save['model'] = self.model
                metadata_path = filepath

            with open(metadata_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            logger.info("Model data saved to %s", metadata_path)
            if self.classifier_type == ClassifierType.MISTRAL and model_dir is not None:
                logger.info("Mistral weights/tokenizer saved in %s", model_dir)

        except Exception as e:
            logger.error("Save error: %s", e)
            raise ValueError(f"Failed to save model: {e}")

    @classmethod
    def load(cls, filepath: str | None = None, classifier_type_hint: ClassifierType | None = None) -> 'KnowledgeGraphClassifier':
        if filepath is None and classifier_type_hint:
            filepath = f'../data/trained/model-{classifier_type_hint.name}.pkl'
        elif filepath is None:
            raise ValueError("Filepath must be provided if classifier_type_hint is not.")

        if not os.path.exists(filepath):
            raise ValueError(f"Model file/metadata {filepath} does not exist. Please train and save the model first.")

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            cls_type = data.get('classifier_type')
            if not isinstance(cls_type, ClassifierType):
                cls_type = ClassifierType[cls_type]

            instance = cls(classifier_type=cls_type)
            instance.target_label_mapping = data.get('target_label_mapping')
            instance.target_label_inverse_mapping = data.get('target_label_inverse_mapping')
            instance.feature_labels = data.get('feature_labels')
            instance.max_length = data.get('max_length')
            instance.accuracy = data.get('accuracy', 0.0)

            if instance.classifier_type == ClassifierType.MISTRAL:
                model_dir = data.get('model_dir')
                if model_dir is None or not os.path.isdir(model_dir):
                    raise ValueError(f"Saved Mistral model directory '{model_dir}' not found or invalid.")
                logger.info("Loading Mistral model from %s", model_dir)
                instance.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, device_map="auto", torch_dtype=torch.bfloat16
                )
                instance.mistral_tokenizer = AutoTokenizer.from_pretrained(model_dir)
                logger.info("Mistral model loaded.")
            else:
                instance.vectorizer = data.get('vectorizer')
                instance.model = data.get('model')
                if instance.vectorizer is None or instance.model is None:
                    raise ValueError("Classic model pickle missing 'vectorizer' or 'model'.")
                logger.info("Classic model loaded from %s", filepath)
            return instance
        except Exception as e:
            logger.error("Load error: %s", e)
            raise ValueError(f"Failed to load model from {filepath}: {e}")

def save_multiple_models(
    models: dict[str, KnowledgeGraphClassifier],
    training_results: dict[str, Any],
    filepath: str | None = None
) -> None:
    if filepath is None:
        os.makedirs('../data/trained', exist_ok=True)
        filepath = '../data/trained/multiple_models.pkl'
    to_pickle = {'models': {}, 'training_results': training_results}
    for feature, model in models.items():
        meta = {
            'classifier_type': model.classifier_type,
            'target_label_mapping': model.target_label_mapping,
            'target_label_inverse_mapping': model.target_label_inverse_mapping,
            'feature_labels': model.feature_labels,
            'max_length': model.max_length,
            'accuracy': model.accuracy,
        }
        if model.classifier_type == ClassifierType.MISTRAL:
            model_dir = f'../data/trained/mistral_{feature}'
            os.makedirs(model_dir, exist_ok=True)
            model.model.save_pretrained(model_dir)
            if model.mistral_tokenizer:
                model.mistral_tokenizer.save_pretrained(model_dir)
            meta['model_dir'] = model_dir
        else:
            meta['vectorizer'] = model.vectorizer
            meta['model'] = model.model
        to_pickle['models'][feature] = meta
    with open(filepath, 'wb') as f:
        pickle.dump(to_pickle, f)

def load_multiple_models(filepath: str | None = None) -> Tuple[dict[str, KnowledgeGraphClassifier], dict[str, Any]]:
    if filepath is None:
        filepath = '../data/trained/multiple_models.pkl'
    if not os.path.exists(filepath):
        raise ValueError(f"Multiple models file {filepath} does not exist. Please run training and save the models first.")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    models = {}
    for feature, meta in data['models'].items():
        cls_type = meta.get('classifier_type')
        if not isinstance(cls_type, ClassifierType):
            cls_type = ClassifierType[cls_type]
        model = KnowledgeGraphClassifier(classifier_type=cls_type)
        model.target_label_mapping = meta.get('target_label_mapping')
        model.target_label_inverse_mapping = meta.get('target_label_inverse_mapping')
        model.feature_labels = meta.get('feature_labels')
        model.max_length = meta.get('max_length')
        model.accuracy = meta.get('accuracy', 0.0)
        if cls_type == ClassifierType.MISTRAL:
            model_dir = meta.get('model_dir')
            if model_dir is None or not os.path.isdir(model_dir):
                raise ValueError(f"Saved Mistral model directory '{model_dir}' not found or invalid.")
            model.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, device_map="auto", torch_dtype=torch.bfloat16
            )
            model.mistral_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            model.vectorizer = meta.get('vectorizer')
            model.model = meta.get('model')
        models[feature] = model
    return models, data['training_results']