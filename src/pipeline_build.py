from __future__ import annotations

import gc
import logging
import os
import pickle
import re
import warnings
import torch
import numpy as np
import pandas as pd
from collections import Counter
from enum import Enum, auto
from typing import Any, Tuple, NewType, TypeAlias
from joblib import Memory
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging as hf_logging,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

hf_logging.set_verbosity_error()
np.seterr(invalid="ignore")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The parameter 'token_pattern' will not be used*"
)
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FeatureLabels: TypeAlias = str | list[str]
TextData = NewType("TextData", str)


class ClassifierType(Enum):
    SVM = auto()
    NAIVE_BAYES = auto()
    KNN = auto()
    J48 = auto()  # Decision Tree
    MISTRAL = auto()


def is_uri(token: str) -> bool:
    uri_regex = re.compile(
        r"^(?:https?|ftp|file)://[^\s<>'\"`]+$|^www\.[^\s<>'\"`]+$|^[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-zA-Z]{2,}.*$"
    )
    return bool(uri_regex.match(token))


def hybrid_tokenizer(text: str) -> list[str]:
    tokens = text.split()
    processed: list[str] = []
    for tok in tokens:
        if is_uri(tok):
            processed.append(tok)
        else:
            processed.extend(re.findall(r"(?u)\b\w\w+\b", tok))
    return processed


def get_custom_tfidf_vectorizer(**kwargs) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        tokenizer=hybrid_tokenizer,
        token_pattern=None,
        binary=False,
        norm="l2",
        min_df=1,
        max_df=0.85,
        **kwargs
    )


def get_custom_count_vectorizer(**kwargs) -> CountVectorizer:
    return CountVectorizer(
        tokenizer=hybrid_tokenizer,
        token_pattern=None,
        min_df=1,
        max_df=0.85,
        **kwargs
    )


def majority_vote(predictions: list[Any] | str | Tuple[Any, float, str]) -> Any:
    if isinstance(predictions, str):
        return predictions

    filtered = [p for p in predictions if p is not None]
    if not filtered:
        logger.info("No valid predictions available for majority vote.")
        return None

    if all(isinstance(p, (tuple, list)) and len(p) >= 2 for p in filtered):
        weight_sum: dict[Any, float] = {}
        for tup in filtered:
            label = tup[0]
            try:
                w = float(tup[1])
            except Exception:
                w = 1.0
            weight_sum[label] = weight_sum.get(label, 0.0) + w

        if not weight_sum:
            logger.info("No weighted predictions available for majority vote.")
            return None

        best_label, best_weight = max(weight_sum.items(), key=lambda x: x[1])
        logger.info("Weighted vote: '%s' with total weight %.4f", best_label, best_weight)
        return best_label

    simple_labels = [p for p in filtered if not isinstance(p, (tuple, list))]
    if not simple_labels:
        simple_labels = [p[0] for p in filtered]
    most_common = Counter(simple_labels).most_common(1)[0][0]
    logger.info("Majority vote result: '%s'", most_common)
    return most_common


def _predict_category_for_instance(
    models: dict[str, KnowledgeGraphClassifier],
    instance: dict[str, Any]
) -> Any | None:
    votes: list[tuple[Any, float, str]] = []
    for feature_name, model in models.items():
        value = instance.get(feature_name)
        if value is None:
            continue

        if isinstance(value, str):
            v = value.strip()
            if not v:
                continue
            feature_data = [v]
        elif isinstance(value, (list, set, tuple)):
            raw_list = [str(item).strip() for item in value if item is not None]
            feature_data = [s for s in raw_list if s]
            if not feature_data:
                continue
        else:
            v = str(value).strip()
            if not v:
                continue
            feature_data = [v]

        try:
            pred_arr = model.predict(feature_data)
            if len(pred_arr) == 0:
                continue
            pred_label = pred_arr[0]
            score = getattr(model, "accuracy", 0.5)
            votes.append((pred_label, score, feature_name))
        except Exception as err:
            logger.error("Prediction error for feature '%s': %s", feature_name, err)

    if not votes:
        return None

    return majority_vote(votes)


def predict_category_multi(
    models: dict[str, KnowledgeGraphClassifier],
    instance: dict[str, Any] | pd.DataFrame
) -> Any | list[Any | None]:
    if isinstance(instance, pd.DataFrame):
        return instance.apply(
            lambda row: _predict_category_for_instance(models, row.to_dict()),
            axis=1
        ).tolist()

    return _predict_category_for_instance(models, instance)


def remove_empty_rows(frame: pd.DataFrame, labels: str | list[str]) -> pd.DataFrame:
    if isinstance(labels, str):
        labels = [labels]

    df = frame.copy()
    for lbl in labels:
        df = df.dropna(subset=[lbl])
        df = df[df[lbl] != ""]
    return df


def oversample_dataframe(df: pd.DataFrame, target_label: str, max_factor: float = 1.5) -> pd.DataFrame:
    counts = df[target_label].value_counts()
    logger.info("Original class counts: %s", counts.to_dict())
    max_count = counts.max()
    groups: list[pd.DataFrame] = []
    for lbl, grp in df.groupby(target_label):
        desired = min(max_count, int(len(grp) * max_factor))
        if len(grp) < desired:
            sampled = grp.sample(desired, replace=True, random_state=42)
            groups.append(sampled)
        else:
            groups.append(grp)

    if not groups:
        return pd.DataFrame(columns=df.columns)

    oversampled = pd.concat(groups).sample(frac=1.0, random_state=42).reset_index(drop=True)
    new_counts = oversampled[target_label].value_counts()
    logger.info("After oversampling: %s", new_counts.to_dict())
    return oversampled


def train_multiple_models(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    target_label: str = "category",
    classifier_type: ClassifierType = ClassifierType.NAIVE_BAYES,
    oversample: bool = True,  # <-- new flag
) -> Tuple[dict[str, KnowledgeGraphClassifier], dict[str, Any]]:
    """
    For each feature in `feature_columns`, train a separate KnowledgeGraphClassifier.
    If oversample=False, no oversampling will be applied (because balance_classes=False).
    """
    models: dict[str, KnowledgeGraphClassifier] = {}
    training_results: dict[str, Any] = {}

    for feature in tqdm(feature_columns, desc="Training models"):
        df_feat = remove_empty_rows(training_data, [feature, target_label])
        if df_feat.empty:
            logger.info("Skipping '%s': no data available.", feature)
            continue

        if df_feat[target_label].nunique() < 2:
            logger.info(
                "Skipping '%s': only one unique class: %s",
                feature,
                df_feat[target_label].unique()
            )
            continue

        logger.info("Training model for feature '%s' with %d samples", feature, len(df_feat))

        # Pass oversample → balance_classes. If oversample=False, .train() will never call oversample_dataframe.
        model = KnowledgeGraphClassifier(
            classifier_type=classifier_type,
            balance_classes=oversample
        )
        try:
            result = model.train(df_feat, feature, target_label=target_label)
        except Exception as e:
            logger.error("Error training '%s': %s", feature, e)
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
        if state.is_local_process_zero:
            self.pbar = tqdm(total=self.total_steps, desc="Training samples", position=0, leave=True)
            self.current_step = state.global_step

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_local_process_zero and self.pbar is not None:
            step_diff = state.global_step - self.current_step
            if step_diff > 0:
                self.pbar.update(step_diff)
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
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class KnowledgeGraphClassifier:
    def __init__(
        self,
        classifier_type: ClassifierType = ClassifierType.NAIVE_BAYES,
        balance_classes: bool = False,
        vectorizer_type: str = "tfidf",
        feature_labels: FeatureLabels | None = None,
        **kwargs
    ):
        self.classifier_type = classifier_type
        self.balance_classes = balance_classes
        self.vectorizer_type = vectorizer_type
        self.feature_labels = feature_labels
        self.vectorizer = self._init_vectorizer(**kwargs)
        self.model: Any = None
        self.mistral_tokenizer: Any = None
        self.target_label_mapping: dict[Any, int] | None = None
        self.target_label_inverse_mapping: dict[int, Any] | None = None
        self.max_length: int | None = None
        self.accuracy: float = 0.0

    def _init_vectorizer(self, **kwargs) -> TfidfVectorizer | CountVectorizer:
        if self.vectorizer_type.lower() == "tfidf":
            return get_custom_tfidf_vectorizer(**kwargs)
        return get_custom_count_vectorizer(**kwargs)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\[\s*\]", "", text)
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\{\s*\}", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _prepare_features(self, frame: pd.DataFrame, feature_labels: FeatureLabels) -> pd.Series:
        import ast
        import json

        def parse_val(val: Any) -> str:
            if isinstance(val, (list, tuple, set)):
                return " ".join(str(x) for x in val)
            if isinstance(val, str):
                s = val.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, (list, tuple, set)):
                            return " ".join(str(x) for x in parsed)
                    except Exception:
                        pass
                    try:
                        parsed = json.loads(s.replace("'", '"'))
                        if isinstance(parsed, (list, tuple, set)):
                            return " ".join(str(x) for x in parsed)
                    except Exception:
                        pass
                return s
            return str(val)

        if isinstance(feature_labels, (list, tuple)):
            df_copy = frame.loc[:, feature_labels].copy()
            for col in df_copy.columns:
                df_copy[col] = df_copy[col].map(parse_val)
            concatenated = df_copy.agg(" ".join, axis=1)
            return concatenated

        single_col = frame[feature_labels].map(parse_val)
        return single_col

    def train(
        self,
        frame: pd.DataFrame,
        feature_labels: FeatureLabels,
        target_label: str = "category",
        max_length: int = 8000000
    ) -> dict[str, Any]:
        if self.classifier_type == ClassifierType.MISTRAL:
            return self.train_mistral(frame, feature_labels, target_label=target_label, max_length=max_length)

        self.feature_labels = feature_labels
        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]
        unique_labels = sorted(data_y.unique())
        if len(unique_labels) < 2:
            raise ValueError(f"Need ≥2 unique classes in '{target_label}', found {len(unique_labels)}")

        self.target_label_mapping = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self.target_label_inverse_mapping = {idx: lbl for lbl, idx in self.target_label_mapping.items()}
        y_numeric = data_y.map(self.target_label_mapping).values

        if len(data_x) != len(y_numeric):
            raise ValueError("Feature length & target length mismatch after preparation.")
        if len(data_x) == 0:
            raise ValueError("No data available for training after preparation.")

        X_train, X_temp, y_train, y_temp = train_test_split(
            data_x, y_numeric, test_size=0.20, random_state=42, stratify=y_numeric
        )

        X_val: pd.Series | list[str] = []
        y_val: np.ndarray | list[int] = []
        X_test: pd.Series | list[str] = []
        y_test: np.ndarray | list[int] = []

        if len(X_temp) > 1 and len(set(y_temp)) > 1:
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
            )
        elif len(X_temp) > 0:
            X_val = X_temp
            y_val = y_temp

        if self.balance_classes:
            vc = pd.Series(y_train).value_counts()
            if len(vc) > 1 and (vc.min() / vc.max()) < 0.75:
                logger.info("Unbalanced train set detected; applying oversampling.")
                df_train = pd.DataFrame({"X": X_train, "y": y_train})
                df_train = oversample_dataframe(df_train, "y", max_factor=2.0)
                if not df_train.empty:
                    X_train = df_train["X"]
                    y_train = df_train["y"].values
                    logger.info(
                        "Post‐oversample train counts: %s",
                        pd.Series(y_train).value_counts().to_dict()
                    )
                else:
                    logger.warning("Oversample returned empty DataFrame; skipping oversample.")

        clf_type = self.classifier_type
        if clf_type == ClassifierType.SVM:
            estimator = svm.SVC(
                probability=True, random_state=42, class_weight="balanced"
            )
            param_dist = {
                "classifier__C": np.logspace(-2, 2, 8),
                "classifier__kernel": ["linear", "rbf", "sigmoid"],
                "classifier__gamma": ["scale", "auto"],
            }
        elif clf_type == ClassifierType.NAIVE_BAYES:
            estimator = MultinomialNB()
            param_dist = {
                "classifier__alpha": np.linspace(0.1, 2, 7),
                "classifier__fit_prior": [True, False],
            }
        elif clf_type == ClassifierType.KNN:
            estimator = KNeighborsClassifier()
            param_dist = {
                "classifier__n_neighbors": list(range(2, 7)),
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            }
        elif clf_type == ClassifierType.J48:
            estimator = DecisionTreeClassifier(class_weight="balanced")
            param_dist = {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__splitter": ["best", "random"],
                "classifier__max_depth": [2, 5, 10, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 5],
                "classifier__max_features": ["sqrt", "log2", None],
            }
        else:
            raise ValueError("Unsupported classic classifier type.")


        memory = Memory("../data/trained/cache", verbose=0)
        pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", estimator),
        ], memory=memory)

        if len(X_val) > 0:
            X_combined = pd.concat([pd.Series(X_train), pd.Series(X_val)], ignore_index=True)
            y_combined = np.concatenate([np.array(y_train), np.array(y_val)], axis=0)
        else:
            X_combined = X_train
            y_combined = y_train

        def total_combinations(d: dict[str, list[Any]]) -> int:
            prod = 1
            for choices in d.values():
                prod *= len(choices)
            return prod

        total_poss = total_combinations(param_dist)
        n_iter = min(20, total_poss)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=2,  # as requested
            scoring="f1_weighted",
            verbose=1,
            n_jobs=-1,
            error_score=0.0,
            random_state=42,
            return_train_score=False
        )

        logger.info(
            "Running RandomizedSearchCV with up to %d combos (out of %d total)",
            n_iter,
            total_poss
        )
        search.fit(X_combined, y_combined)
        best_params = search.best_params_
        logger.info("Best hyperparameters: %s", best_params)

        final_model: Pipeline = search.best_estimator_

        if len(X_test) > 0:
            y_pred_test = final_model.predict(X_test)
            final_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
            final_acc = accuracy_score(y_test, y_pred_test)
            conf_mat = confusion_matrix(y_test, y_pred_test)
            report = classification_report(
                y_test,
                y_pred_test,
                target_names=[str(self.target_label_inverse_mapping[i]) for i in range(len(self.target_label_inverse_mapping))],
                zero_division=0
            )
        else:
            final_f1 = 0.0
            final_acc = 0.0
            conf_mat = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
            report = "No hold‐out test set was available to evaluate."

        self.accuracy = final_f1
        self.model = final_model

        logger.info("\n%s", report)
        logger.info("Final F1: %.4f, Accuracy: %.4f", final_f1, final_acc)

        print("\nConfusion Matrix:\n", conf_mat)
        if len(X_test) > 0:
            print("\n=== Misclassified Test Examples ===")
            for txt, true_idx, pred_idx in zip(X_test, y_test, y_pred_test):
                if true_idx != pred_idx:
                    token_list = self.vectorizer.build_tokenizer()(txt)
                    print(f"INPUT TOKENS: {token_list}")
                    print(
                        f"TRUE: {self.target_label_inverse_mapping[true_idx]} | PRED: {self.target_label_inverse_mapping[pred_idx]}\n"
                    )

        return {
            "best_params": best_params,
            "final_f1": final_f1,
            "final_accuracy": final_acc,
            "confusion_matrix": conf_mat.tolist(),
            "feature_labels": feature_labels,
            "target_label_mapping": self.target_label_mapping,
            "classifier_type": self.classifier_type,
            "classification_report": report
        }

    def train_mistral(
        self,
        frame: pd.DataFrame,
        feature_labels: FeatureLabels,
        target_label: str = "category",
        max_length: int = 512,
        qlora_r: int = 32,
        qlora_alpha: int = 64,
        batch_size: int = 8,
        epochs: int = 15
    ) -> dict[str, Any] | None:
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Mistral training.")
        if LoraConfig is None or get_peft_model is None or prepare_model_for_kbit_training is None:
            raise ImportError("peft is not installed. Run `pip install peft` first.")

        frame = frame.reset_index(drop=True)
        frame = frame.dropna(subset=[target_label])
        frame = frame[frame[target_label] != ""]

        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]
        unique_labels = sorted(data_y.unique())
        if len(unique_labels) < 2:
            raise ValueError(f"Need ≥2 unique classes in '{target_label}', found {len(unique_labels)}")

        self.target_label_mapping = {str(lbl): idx for idx, lbl in enumerate(unique_labels)}
        self.target_label_inverse_mapping = {idx: str(lbl) for idx, lbl in enumerate(unique_labels)}
        num_classes = len(unique_labels)

        texts = [self._clean_text(str(t)) for t in data_x.tolist()]
        y_numeric = [self.target_label_mapping[str(lbl)] for lbl in data_y.tolist()]

        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, y_numeric, test_size=0.20, random_state=42, stratify=y_numeric
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
        enc_train = self.mistral_tokenizer(
            X_train, truncation=True, padding="max_length", max_length=self.max_length
        )
        train_dataset = HFDataset(enc_train, y_train)

        enc_val = (
            self.mistral_tokenizer(X_val, truncation=True, padding="max_length", max_length=self.max_length)
            if X_val
            else None
        )
        val_dataset = HFDataset(enc_val, y_val) if X_val else None

        enc_test = (
            self.mistral_tokenizer(X_test, truncation=True, padding="max_length", max_length=self.max_length)
            if X_test
            else None
        )
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

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_classes,
            quantization_config=quant_config,
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

        lora_cfg = LoraConfig(
            r=qlora_r,
            lora_alpha=qlora_alpha,
            lora_dropout=0.01,
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=batch_size,
            logging_steps=500,
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
                "f1": f1_score(labels, preds, average="weighted", zero_division=0)
            }

        total_examples = len(train_dataset)
        callbacks = [TqdmDataCallback(total_steps=total_examples)]
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
        feature_labels: FeatureLabels | None = None
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained/loaded. Call `.train()` or `.load()` first.")

        if feature_labels is None:
            if self.feature_labels is None:
                raise ValueError("feature_labels must be provided if not set during training.")
            feature_labels = self.feature_labels

        if isinstance(data, pd.DataFrame):
            texts = self._prepare_features(data, feature_labels).tolist()
        elif isinstance(data, pd.Series):
            if isinstance(feature_labels, list) and len(feature_labels) > 1:
                raise ValueError("Cannot feed pd.Series with multiple feature_labels; provide a DataFrame instead.")
            temp_df = data.to_frame(name=feature_labels if isinstance(feature_labels, str) else feature_labels[0])
            texts = self._prepare_features(temp_df, feature_labels).tolist()
        elif isinstance(data, list):
            texts = [self._clean_text(str(item)) for item in data]
        elif isinstance(data, str):
            texts = [self._clean_text(data)]
        else:
            raise ValueError("Unsupported data type for predict. Use str, list[str], pd.Series, or pd.DataFrame.")

        if not texts:
            return np.array([])

        if self.classifier_type == ClassifierType.MISTRAL:
            import torch
            from tqdm import tqdm

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for Mistral inference.")
            if self.mistral_tokenizer is None:
                raise ValueError("Mistral tokenizer not found. Did you `.load()` the model?")
            if self.target_label_inverse_mapping is None:
                raise ValueError("Label mapping missing. Did you `.load()` the model?")

            results: list[int] = []
            batch_size = 16
            self.model.eval()
            for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", leave=False):
                batch_texts = texts[i : i + batch_size]
                inputs = self.mistral_tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.model.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                results.extend(preds)

            return np.array([self.target_label_inverse_mapping.get(int(idx), None) for idx in results])

        if not hasattr(self.model, "predict"):
            raise ValueError("Classic model has no `.predict` method.")

        try:
            preds_numeric = self.model.predict(texts)
        except Exception as e:
            logger.error("Classic prediction error: %s", e)
            raise ValueError(f"Prediction failed for classic model: {e}")

        if self.target_label_inverse_mapping is not None:
            return np.array([self.target_label_inverse_mapping.get(int(i), None) for i in preds_numeric])

        return np.array(preds_numeric)

    def save(self, filepath: str | None = None) -> None:
        if self.model is None:
            raise ValueError("Nothing to save; model is None.")

        if filepath is None:
            os.makedirs("../data/trained", exist_ok=True)
            filepath = f"../data/trained/model-{self.classifier_type.name}.pkl"

        data_to_save: dict[str, Any] = {
            "classifier_type": self.classifier_type,
            "target_label_mapping": self.target_label_mapping,
            "target_label_inverse_mapping": self.target_label_inverse_mapping,
            "feature_labels": self.feature_labels,
            "max_length": self.max_length,
            "accuracy": self.accuracy
        }

        try:
            if self.classifier_type == ClassifierType.MISTRAL:
                model_dir = filepath.replace(".pkl", "")
                os.makedirs(model_dir, exist_ok=True)
                self.model.save_pretrained(model_dir)
                if self.mistral_tokenizer:
                    self.mistral_tokenizer.save_pretrained(model_dir)
                data_to_save["model_dir"] = model_dir
                metadata_path = filepath
            else:
                data_to_save["vectorizer"] = self.vectorizer
                data_to_save["model"] = self.model
                metadata_path = filepath

            with open(metadata_path, "wb") as f:
                pickle.dump(data_to_save, f)
            logger.info("Saved metadata to %s", metadata_path)
            if self.classifier_type == ClassifierType.MISTRAL:
                logger.info("Saved Mistral weights/tokenizer under %s", model_dir)
        except Exception as e:
            logger.error("Save error: %s", e)
            raise ValueError(f"Failed to save model: {e}")

    @classmethod
    def load(
        cls,
        filepath: str | None = None,
        classifier_type_hint: ClassifierType | None = None
    ) -> KnowledgeGraphClassifier:
        if filepath is None and classifier_type_hint:
            filepath = f"../data/trained/model-{classifier_type_hint.name}.pkl"
        elif filepath is None:
            raise ValueError("Must provide filepath if no classifier_type_hint given.")

        if not os.path.exists(filepath):
            raise ValueError(f"Model metadata {filepath} not found. Train & save first.")

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            cls_type = data.get("classifier_type")
            if not isinstance(cls_type, ClassifierType):
                cls_type = ClassifierType[cls_type]

            instance = cls(classifier_type=cls_type)
            instance.target_label_mapping = data.get("target_label_mapping")
            instance.target_label_inverse_mapping = data.get("target_label_inverse_mapping")
            instance.feature_labels = data.get("feature_labels")
            instance.max_length = data.get("max_length")
            instance.accuracy = data.get("accuracy", 0.0)

            if instance.classifier_type == ClassifierType.MISTRAL:
                model_dir = data.get("model_dir")
                if model_dir is None or not os.path.isdir(model_dir):
                    raise ValueError(f"Mistral model_dir '{model_dir}' missing/invalid.")
                logger.info("Loading Mistral from %s", model_dir)
                instance.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, device_map="auto", torch_dtype=torch.bfloat16
                )
                instance.mistral_tokenizer = AutoTokenizer.from_pretrained(model_dir)
                logger.info("Loaded Mistral successfully.")
            else:
                instance.vectorizer = data.get("vectorizer")
                instance.model = data.get("model")
                if instance.vectorizer is None or instance.model is None:
                    raise ValueError("Classic model missing vectorizer/model in pickle.")
                logger.info("Loaded classic model from %s", filepath)

            return instance

        except Exception as e:
            logger.error("Load error: %s", e)
            raise ValueError(f"Failed to load model: {e}")


def save_multiple_models(
    models: dict[str, KnowledgeGraphClassifier],
    training_results: dict[str, Any],
    filepath: str | None = None
) -> None:
    if filepath is None:
        os.makedirs("../data/trained", exist_ok=True)
        filepath = "../data/trained/multiple_models.pkl"

    to_pickle: dict[str, Any] = {"models": {}, "training_results": training_results}
    for feature, model in models.items():
        meta: dict[str, Any] = {
            "classifier_type": model.classifier_type,
            "target_label_mapping": model.target_label_mapping,
            "target_label_inverse_mapping": model.target_label_inverse_mapping,
            "feature_labels": model.feature_labels,
            "max_length": model.max_length,
            "accuracy": model.accuracy,
        }

        if model.classifier_type == ClassifierType.MISTRAL:
            model_dir = f"../data/trained/mistral_{feature}"
            os.makedirs(model_dir, exist_ok=True)
            model.model.save_pretrained(model_dir)
            if model.mistral_tokenizer:
                model.mistral_tokenizer.save_pretrained(model_dir)
            meta["model_dir"] = model_dir
        else:
            meta["vectorizer"] = model.vectorizer
            meta["model"] = model.model

        to_pickle["models"][feature] = meta

    with open(filepath, "wb") as f:
        pickle.dump(to_pickle, f)


def load_multiple_models(
    filepath: str | None = None
) -> Tuple[dict[str, KnowledgeGraphClassifier], dict[str, Any]]:
    if filepath is None:
        filepath = "../data/trained/multiple_models.pkl"
    if not os.path.exists(filepath):
        raise ValueError(f"multiple_models file '{filepath}' not found. Train & save first.")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    loaded_models: dict[str, KnowledgeGraphClassifier] = {}
    for feature, meta in data["models"].items():
        cls_type = meta.get("classifier_type")
        if not isinstance(cls_type, ClassifierType):
            cls_type = ClassifierType[cls_type]

        model = KnowledgeGraphClassifier(classifier_type=cls_type)
        model.target_label_mapping = meta.get("target_label_mapping")
        model.target_label_inverse_mapping = meta.get("target_label_inverse_mapping")
        model.feature_labels = meta.get("feature_labels")
        model.max_length = meta.get("max_length")
        model.accuracy = meta.get("accuracy", 0.0)

        if cls_type == ClassifierType.MISTRAL:
            mdl_dir = meta.get("model_dir")
            if mdl_dir is None or not os.path.isdir(mdl_dir):
                raise ValueError(f"Mistral directory '{mdl_dir}' for feature '{feature}' missing/invalid.")
            model.model = AutoModelForSequenceClassification.from_pretrained(
                mdl_dir, device_map="auto", torch_dtype=torch.bfloat16
            )
            model.mistral_tokenizer = AutoTokenizer.from_pretrained(mdl_dir)
        else:
            model.vectorizer = meta.get("vectorizer")
            model.model = meta.get("model")

        loaded_models[feature] = model

    return loaded_models, data["training_results"]
