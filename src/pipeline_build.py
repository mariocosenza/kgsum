from __future__ import annotations

import logging
import os
import pickle
import re
import warnings
from enum import Enum, auto
from typing import Any, Tuple, NewType, TypeAlias, Self

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, make_scorer
from collections import Counter
from sklearn.pipeline import Pipeline

from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    logging as hf_logging,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch
import gc

FeatureLabels: TypeAlias = str | list[str]
TextData = NewType('TextData', str)

hf_logging.set_verbosity_error()
np.seterr(invalid='ignore')

warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'token_pattern' will not be used*")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassifierType(Enum):
    SVM = auto()
    NAIVE_BAYES = auto()
    KNN = auto()
    ROBERTA = auto()


def majority_vote(predictions: list[Any]) -> Any:
    # If predictions is actually a string, treat it as a single prediction.
    if isinstance(predictions, str):
        return predictions

    if not predictions:
        logger.info("No predictions available for majority vote.")
        return None

    # Check if votes are weighted with feature info.
    if isinstance(predictions[0], (tuple, list)):
        if len(predictions[0]) == 3 and isinstance(predictions[0][1], (float, int)):
            candidate_info = {}  # candidate -> {'features': [(feature, f1), ...], 'score': total_score}
            for pred, f1, feature in predictions:
                if pred in candidate_info:
                    candidate_info[pred]['features'].append((feature, f1))
                    candidate_info[pred]['score'] += f1
                else:
                    candidate_info[pred] = {'features': [(feature, f1)], 'score': f1}
            # Log candidate contributions.
            for candidate, info in candidate_info.items():
                features_str = ", ".join(f"{feat} (F1={f1:.2f})" for feat, f1 in info['features'])
                logger.info("Candidate '%s': features -> %s, total F1 sum = %.4f", candidate, features_str, info['score'])
            chosen, chosen_info = max(candidate_info.items(), key=lambda x: x[1]['score'])
            logger.info("Weighted majority vote result: chosen candidate '%s' with total F1 sum %.4f", chosen, chosen_info['score'])
            return chosen
        elif len(predictions[0]) == 2 and isinstance(predictions[0][1], (float, int)):
            # Weighted votes without feature info: use sum.
            products = {}
            for pred, f1 in predictions:
                products[pred] = products.get(pred, 0) + f1
            chosen, score = max(products.items(), key=lambda x: x[1])
            logger.info("Weighted majority vote result: label '%s' with total F1 sum %.4f", chosen, score)
            return chosen
        else:
            filtered_predictions = [p for p in predictions if p is not None]
            if not filtered_predictions:
                logger.info("No valid predictions available for majority vote.")
                return None
            chosen = Counter(filtered_predictions).most_common(1)[0][0]
            logger.info("Standard majority vote result: label '%s'", chosen)
            return chosen
    else:
        filtered_predictions = [p for p in predictions if p is not None]
        if not filtered_predictions:
            logger.info("No valid predictions available for majority vote.")
            return None
        chosen = Counter(filtered_predictions).most_common(1)[0][0]
        logger.info("Standard majority vote result: label '%s'", chosen)
        return chosen


def _predict_category_for_instance(
    models: dict[str, KnowledgeGraphClassifier],
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
            # Use predict_batched if the classifier is ROBERTA
            if model.classifier_type == ClassifierType.ROBERTA:
                pred = model.predict_batched(feature_data)[0]
            else:
                pred = model.predict(feature_data)[0]
            score = getattr(model, "accuracy", 0.5)
            votes.append((pred, score, feature))
        except Exception as err:
            logger.error("Prediction error for %s: %s", feature, err)
    return majority_vote(votes)


def predict_category_multi(
    models: dict[str, KnowledgeGraphClassifier],
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


def oversample_dataframe(df: pd.DataFrame, target_label: str, max_factor: float = 1.1) -> pd.DataFrame:
    counts = df[target_label].value_counts()
    max_count = counts.max()
    groups = []
    # return df
    for label, group in df.groupby(target_label):
        target_count = min(max_count, int(len(group) * max_factor))
        if len(group) < target_count:
            groups.append(group.sample(target_count, replace=True, random_state=42))
        else:
            groups.append(group)
    return pd.concat(groups).sample(frac=1, random_state=42).reset_index(drop=True)


def train_multiple_models(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    target_label: str = 'category',
    classifier_type: ClassifierType = ClassifierType.SVM
) -> Tuple[dict[str, KnowledgeGraphClassifier], dict[str, Any]]:
    models: dict[str, KnowledgeGraphClassifier] = {}
    training_results: dict[str, Any] = {}
    for feature in feature_columns:
        df_feature = remove_empty_rows(training_data, [feature, target_label])
        if df_feature.empty:
            logger.info("Skipping '%s': no data available.", feature)
            continue
        if df_feature[target_label].nunique() < 2:
            logger.info("Skipping '%s': only one unique class: %s", feature, df_feature[target_label].unique())
            continue
        counts = df_feature[target_label].value_counts()
        if (counts.min() / counts.max()) < 0.75:
            logger.info("Feature '%s' unbalanced; applying oversampling.", feature)
            df_feature = oversample_dataframe(df_feature, target_label, max_factor=2.0)
        logger.info("Training model for '%s' with %s examples.", feature, len(df_feature))
        model = KnowledgeGraphClassifier(classifier_type=classifier_type, balance_classes=True)
        try:
            result = model.train(df_feature, feature, target_label=target_label)
        except ValueError as err:
            logger.info("Skipping '%s': %s", feature, err)
            continue
        except Exception as err:
            logger.error("Error training model for '%s': %s", feature, err)
            continue
        models[feature] = model
        training_results[feature] = result
    return models, training_results


class RobertaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


class KnowledgeGraphClassifier:
    def __init__(self, classifier_type: ClassifierType = ClassifierType.SVM, balance_classes: bool = True) -> None:
        self.classifier_type = classifier_type
        self.balance_classes = balance_classes
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.custom_tokenizer,
            lowercase=True,
            stop_words=None,
            max_features=1000000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
            use_idf=True,
            norm='l2'
        )
        self.model: GridSearchCV | torch.nn.Module | RobertaForSequenceClassification | None = None
        self.roberta_tokenizer: RobertaTokenizer | None = None
        self.target_label_mapping: dict[Any, int] | None = None
        self.target_label_inverse_mapping: dict[int, Any] | None = None
        self.feature_labels: FeatureLabels | None = None
        self.max_length: int | None = None
        self.accuracy: float = 0.5

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\[\s*\]', '', text)
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\{\s*\}', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _process_input(self, data: Any, feature_labels: FeatureLabels | None) -> Any:
        if feature_labels is None:
            feature_labels = self.feature_labels
            if feature_labels is None:
                raise ValueError("No feature_labels provided or stored during training")
        if isinstance(data, pd.DataFrame):
            return self._prepare_features(data, feature_labels)
        elif isinstance(data, pd.Series):
            return self._prepare_features(data.to_frame().T, feature_labels)
        else:
            return self._clean_text(data) if isinstance(data, str) else data

    def _vectorize(self, data: Any, feature_labels: FeatureLabels | None) -> Any:
        # This method is used in the RoBERTa branch.
        processed = self._process_input(data, feature_labels)
        if self.classifier_type == ClassifierType.ROBERTA:
            texts = [processed] if isinstance(processed, str) else processed
            texts = [self._clean_text(text) for text in texts]
            encoding = self.roberta_tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return encoding
        else:
            if isinstance(processed, str):
                processed = [processed]
            return processed

    def _get_param_grid(self) -> dict[str, list[Any]] | list[dict[str, Any]]:
        if self.classifier_type == ClassifierType.SVM:
            if self.balance_classes:
                return [
                    {"C": [0.01, 0.05, 0.1, 0.5, 1], "kernel": ["linear"], "class_weight": ["balanced"]},
                    {"C": [0.01, 0.05, 0.1, 0.5, 1], "kernel": ["rbf"], "gamma": ["scale", 0.01, 0.1],
                     "class_weight": ["balanced"]},
                    {"C": [0.01, 0.05, 0.1, 0.5, 1], "kernel": ["poly"], "degree": [2], "gamma": ["scale", 0.01],
                     "class_weight": ["balanced"]}
                ]
            else:
                return [
                    {"C": [0.01, 0.05, 0.1, 0.5, 1], "kernel": ["linear"], "class_weight": [None, "balanced"]},
                    {"C": [0.01, 0.05, 0.1, 0.5, 1], "kernel": ["rbf"], "gamma": ["scale", 0.01, 0.1],
                     "class_weight": [None, "balanced"]},
                    {"C": [0.01, 0.05, 0.1, 0.5, 1], "kernel": ["poly"], "degree": [2], "gamma": ["scale", 0.01],
                     "class_weight": [None, "balanced"]}
                ]
        elif self.classifier_type == ClassifierType.NAIVE_BAYES:
            return  {
                "alpha": [
                    0.1, 0.2, 0.3, 0.4, 0.5,
                    0.6, 0.7, 0.8, 0.9, 1.0,
                    1.2, 1.4, 1.6, 1.8, 2.0,
                    2.5, 3.0, 3.5, 4.0, 4.5,
                    5.0, 5.5, 6.0, 6.5, 7.0,
                    7.5, 8.0, 8.5, 9.0, 10.0
                ],
                "fit_prior": [True, False]
            }
        elif self.classifier_type == ClassifierType.KNN:
            return {
                "n_neighbors": [11, 15, 21, 25, 31],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
                "leaf_size": [20, 30, 40]
            }
        return {}

    def custom_tokenizer(self, text: str) -> list[str]:
        return self._clean_text(text).split()

    def _prepare_features(self, frame: pd.DataFrame, feature_labels: FeatureLabels) -> pd.Series:
        if isinstance(feature_labels, str):
            features = frame[feature_labels].fillna('').astype(str)
        else:
            features = frame[feature_labels[0]].fillna('').astype(str)
            for label in feature_labels[1:]:
                features += " " + frame[label].fillna('').astype(str)
        return features.apply(self._clean_text)

    def _freeze_roberta_layers(self, num_layers: int) -> None:
        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.model.roberta.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def _get_base_estimator(self):
        if self.classifier_type == ClassifierType.SVM:
            return svm.SVC(probability=True, random_state=42)
        elif self.classifier_type == ClassifierType.NAIVE_BAYES:
            return MultinomialNB()
        elif self.classifier_type == ClassifierType.KNN:
            return KNeighborsClassifier()
        else:
            raise ValueError(f"Base estimator not defined for classifier type: {self.classifier_type}")

    def _get_cv_strategy(self, data_y: pd.Series, cv_folds: int) -> StratifiedKFold:
        return StratifiedKFold(n_splits=cv_folds, random_state=42, shuffle=True)

    def train_roberta(
        self,
        frame: pd.DataFrame,
        feature_labels: FeatureLabels,
        target_label: str = 'category',
        max_length: int = 512,
        freeze_layers: int | None = 6,
        num_train_epochs: int = 5,
        batch_size: int = 8,  # Reduced default batch size
        weight_decay: float = 0.01,
    ) -> dict[str, Any]:
        self.feature_labels = feature_labels
        frame = frame.reset_index(drop=True)
        frame = frame.dropna(subset=[target_label])
        frame = frame[frame[target_label] != '']
        if self.balance_classes:
            counts = frame[target_label].value_counts()
            if (counts.min() / counts.max()) < 0.75:
                logger.info("Unbalanced classes; applying oversampling.")
                frame = oversample_dataframe(frame, target_label, max_factor=2.0)

        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]
        unique_labels = sorted(data_y.unique())
        self.target_label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.target_label_inverse_mapping = {idx: label for label, idx in self.target_label_mapping.items()}
        if data_y.nunique() < 2:
            raise ValueError(f"Only one unique class: {data_y.unique()}")

        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length
        texts = [self._clean_text(text) for text in data_x.tolist()]
        encodings = self.roberta_tokenizer(texts, truncation=True, padding=True, max_length=self.max_length)
        y_encoded = data_y.map(self.target_label_mapping).tolist()

        dataset = RobertaDataset(encodings, y_encoded)
        total_len = len(dataset)
        train_size = int(0.9 * total_len)
        val_size = total_len - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.roberta_tokenizer)
        num_classes = len(unique_labels)
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        if freeze_layers is not None:
            self._freeze_roberta_layers(freeze_layers)

        training_args = TrainingArguments(
            output_dir="../data/results",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            disable_tqdm=False,
            seed=42,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=True  # Mixed precision enabled
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
            return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average='weighted')}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.roberta_tokenizer,  # Replaces deprecated tokenizer parameter.
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        # Explicitly clear GPU memory after training to help avoid OOM later
        torch.cuda.empty_cache()
        gc.collect()

        eval_results = trainer.evaluate()
        final_f1 = eval_results.get("eval_f1", 0.0)
        final_acc = eval_results.get("eval_accuracy", 0.0)
        self.accuracy = final_f1
        logger.info("Final RoBERTa F1: %.4f, Accuracy: %.4f", final_f1, final_acc)
        training_history = {"training_args": training_args.to_dict(), "eval_results": eval_results}
        return {
            'history': training_history,
            'final_f1': final_f1,
            'final_accuracy': final_acc,
            'num_classes': num_classes,
            'max_length': self.max_length,
            'target_label_mapping': self.target_label_mapping,
            'feature_labels': feature_labels,
            'classifier_type': self.classifier_type.name,
        }

    def train(
            self,
            frame: pd.DataFrame,
            feature_labels: FeatureLabels,
            target_label: str = 'category',
            cv_folds: int = 2,
            max_length: int = 512,
            freeze_layers: int | None = 6,
    ) -> dict[str, Any]:
        if self.classifier_type == ClassifierType.ROBERTA:
            return self.train_roberta(
                frame=frame,
                feature_labels=feature_labels,
                target_label=target_label,
                max_length=max_length,
                freeze_layers=freeze_layers,
            )
        else:
            # Set the feature_labels for future use (e.g., in predict())
            self.feature_labels = feature_labels
            data_x = self._prepare_features(frame, feature_labels)
            data_y = frame[target_label]
            unique_labels = sorted(data_y.unique())
            self.target_label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            self.target_label_inverse_mapping = {idx: label for label, idx in self.target_label_mapping.items()}

            # Preliminary check: ensure the vectorizer can generate features.
            try:
                x_temp = self.vectorizer.fit_transform(data_x)
                if x_temp.shape[1] == 0:
                    raise ValueError("No features generated during vectorization.")
            except Exception as e:
                logger.error("Vectorization error: %s", e)
                raise ValueError(f"Vectorization failed: {e}")

            # Create a pipeline to ensure the vectorizer is fit within each cross-validation fold.
            pipeline = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self._get_base_estimator())
            ])

            # Modify parameter grid keys to work with the pipeline (prefix with 'classifier__').
            param_grid = self._get_param_grid()
            if isinstance(param_grid, list):
                modified_param_grid = []
                for grid_dict in param_grid:
                    modified_dict = {"classifier__" + k: v for k, v in grid_dict.items()}
                    modified_param_grid.append(modified_dict)
            elif isinstance(param_grid, dict):
                modified_param_grid = {"classifier__" + k: v for k, v in param_grid.items()}
            else:
                modified_param_grid = param_grid

            y_encoded = data_y.map(self.target_label_mapping).values
            scorer = make_scorer(f1_score, average='weighted', zero_division=0)
            cv_strategy = StratifiedKFold(n_splits=cv_folds, random_state=67, shuffle=True)
            try:
                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=modified_param_grid,
                    cv=cv_strategy,
                    scoring=scorer,
                    return_train_score=True,
                    verbose=1,
                    n_jobs=-1,
                    error_score=np.nan,
                )
                grid.fit(data_x, y_encoded)
            except Exception as e:
                logger.error("Grid search error: %s", e)
                raise ValueError(f"Grid search failed: {e}")
            try:
                y_pred = grid.predict(data_x)
                final_f1 = f1_score(y_encoded, y_pred, average='weighted')
                final_acc = accuracy_score(y_encoded, y_pred)
                self.accuracy = final_f1
                conf_mat = confusion_matrix(y_encoded, y_pred)
                logger.info("Final Model F1: %.4f, Accuracy: %.4f", final_f1, final_acc)
            except Exception as e:
                logger.error("Evaluation error: %s", e)
                final_f1, final_acc, conf_mat = np.nan, np.nan, None
            self.model = grid
            return {
                'best_params': grid.best_params_,
                'final_f1': final_f1,
                'final_accuracy': final_acc,
                'confusion_matrix': conf_mat,
                'feature_labels': feature_labels,
                'target_label_mapping': self.target_label_mapping,
                'classifier_type': self.classifier_type.name,
            }

    def predict(
        self,
        data: str | list[str] | pd.Series | pd.DataFrame,
        feature_labels: FeatureLabels | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if self.classifier_type == ClassifierType.ROBERTA:
            # For smaller inputs, you may use this method;
            # however, if you encounter memory issues with large batches, consider using predict_batched.
            x_vectorized = self._vectorize(data, feature_labels)
            try:
                self.model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x_vectorized = {k: v.to(device) for k, v in x_vectorized.items()}
                with torch.no_grad():
                    outputs = self.model(**x_vectorized)
                logits = outputs.logits
                num_labels = self.model.config.num_labels
                numerical_preds = (torch.argmax(logits, dim=1).cpu().numpy()
                                   if num_labels > 1
                                   else (torch.sigmoid(logits).cpu().numpy().flatten() > 0.5).astype(int))
            except Exception as e:
                logger.error("Prediction error: %s", e)
                raise ValueError(f"Prediction failed: {e}")
        else:
            # For non-ROBERTA classifiers (using pipeline), pass raw text directly.
            if feature_labels is None and self.feature_labels:
                feature_labels = self.feature_labels
            if isinstance(data, pd.DataFrame):
                raw_texts = self._prepare_features(data, feature_labels).tolist()
            elif isinstance(data, pd.Series):
                raw_texts = self._prepare_features(data.to_frame().T, feature_labels).tolist()
            elif isinstance(data, list):
                raw_texts = [self._clean_text(item) for item in data]
            elif isinstance(data, str):
                raw_texts = [self._clean_text(data)]
            else:
                raise ValueError("Unsupported data type for prediction.")
            try:
                numerical_preds = self.model.predict(raw_texts)
            except Exception as e:
                logger.error("Prediction error: %s", e)
                raise ValueError(f"Prediction failed: {e}")
        if self.target_label_inverse_mapping:
            return np.array([self.target_label_inverse_mapping[int(idx)] for idx in numerical_preds])
        return numerical_preds

    def predict_batched(
        self,
        data: str | list[str] | pd.Series | pd.DataFrame,
        batch_size: int = 8,
        feature_labels: FeatureLabels | None = None,
    ) -> np.ndarray:
        """
        A new batched prediction method for the RoBERTa branch.
        Use this method for large datasets to avoid large tensor allocations.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if self.classifier_type != ClassifierType.ROBERTA:
            # For non-ROBERTA models, fall back to the standard predict().
            return self.predict(data, feature_labels)
        x_vectorized = self._vectorize(data, feature_labels)
        # Assume all tensors have the same number of examples.
        keys = list(x_vectorized.keys())
        from torch.utils.data import TensorDataset, DataLoader
        # Create a TensorDataset from the encoded inputs; sort keys to maintain consistent order.
        input_tensors = [x_vectorized[k] for k in sorted(x_vectorized.keys())]
        dataset = TensorDataset(*input_tensors)
        loader = DataLoader(dataset, batch_size=batch_size)
        all_logits = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_dict = {}
                for i, key in enumerate(sorted(x_vectorized.keys())):
                    batch_dict[key] = batch[i].to(device)
                outputs = self.model(**batch_dict)
                all_logits.append(outputs.logits.cpu())
        logits = torch.cat(all_logits, dim=0)
        num_labels = self.model.config.num_labels
        if num_labels > 1:
            numerical_preds = torch.argmax(logits, dim=1).numpy()
        else:
            numerical_preds = (torch.sigmoid(logits).numpy().flatten() > 0.5).astype(int)
        if self.target_label_inverse_mapping:
            return np.array([self.target_label_inverse_mapping[int(idx)] for idx in numerical_preds])
        return numerical_preds

    def predict_proba(
        self,
        data: str | list[str] | pd.Series | pd.DataFrame,
        feature_labels: FeatureLabels | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if self.classifier_type == ClassifierType.ROBERTA:
            x_vectorized = self._vectorize(data, feature_labels)
            try:
                self.model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x_vectorized = {k: v.to(device) for k, v in x_vectorized.items()}
                with torch.no_grad():
                    outputs = self.model(**x_vectorized)
                logits = outputs.logits
                return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            except Exception as e:
                logger.error("Predict_proba error: %s", e)
                raise ValueError(f"Probability prediction failed: {e}")
        else:
            if feature_labels is None and self.feature_labels:
                feature_labels = self.feature_labels
            if isinstance(data, pd.DataFrame):
                raw_texts = self._prepare_features(data, feature_labels).tolist()
            elif isinstance(data, pd.Series):
                raw_texts = self._prepare_features(data.to_frame().T, feature_labels).tolist()
            elif isinstance(data, list):
                raw_texts = [self._clean_text(item) for item in data]
            elif isinstance(data, str):
                raw_texts = [self._clean_text(data)]
            else:
                raise ValueError("Unsupported data type for prediction.")
            try:
                return self.model.predict_proba(raw_texts)
            except Exception as e:
                logger.error("Predict_proba error: %s", e)
                raise ValueError(f"Probability prediction failed: {e}")

    def save(self, filepath: str | None = None) -> None:
        if self.model is None:
            raise ValueError("No trained model to save.")
        if filepath is None:
            os.makedirs('../data/trained', exist_ok=True)
            filepath = f'../data/trained/model-{self.classifier_type.name}.pkl'
        data_to_save: dict[str, Any] = {
            'classifier_type': self.classifier_type,
            'target_label_mapping': self.target_label_mapping,
            'target_label_inverse_mapping': self.target_label_inverse_mapping,
            'feature_labels': self.feature_labels,
            'max_length': self.max_length,
        }
        try:
            if self.classifier_type == ClassifierType.ROBERTA:
                model_dir = filepath.replace('.pkl', '')
                os.makedirs(model_dir, exist_ok=True)
                self.model.save_pretrained(model_dir)
                self.roberta_tokenizer.save_pretrained(model_dir)
                data_to_save['model_dir'] = model_dir
            else:
                data_to_save['vectorizer'] = self.vectorizer
                data_to_save['model'] = self.model
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            logger.info("Model saved to %s", filepath)
        except Exception as e:
            logger.error("Save error: %s", e)
            raise ValueError(f"Failed to save model: {e}")

    @classmethod
    def load(cls, filepath: str | None = None, classifier_type: ClassifierType = ClassifierType.SVM) -> Self:
        if filepath is None:
            filepath = f'../data/trained/model-{classifier_type.name}.pkl'
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            cls_type = data.get('classifier_type', classifier_type)
            instance = cls(classifier_type=cls_type)
            instance.target_label_mapping = data.get('target_label_mapping')
            instance.target_label_inverse_mapping = data.get('target_label_inverse_mapping')
            instance.feature_labels = data.get('feature_labels')
            instance.max_length = data.get('max_length')
            if instance.classifier_type == ClassifierType.ROBERTA:
                model_dir = data.get('model_dir')
                if model_dir is None or not os.path.exists(model_dir):
                    raise ValueError("Saved RoBERTa model directory not found.")
                instance.model = RobertaForSequenceClassification.from_pretrained(model_dir)
                instance.roberta_tokenizer = RobertaTokenizer.from_pretrained(model_dir)
            else:
                instance.vectorizer = data.get('vectorizer')
                instance.model = data.get('model')
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
    try:
        with open(filepath, 'wb') as f:
            pickle.dump({'models': models, 'training_results': training_results}, f)
        logger.info("Multiple models saved to %s", filepath)
    except Exception as e:
        logger.error("Saving multiple models error: %s", e)
        raise ValueError(f"Failed to save multiple models: {e}")


def load_multiple_models(filepath: str | None = None) -> Tuple[dict[str, KnowledgeGraphClassifier], dict[str, Any]]:
    if filepath is None:
        filepath = '../data/trained/multiple_models.pkl'
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info("Multiple models loaded from %s", filepath)
        return data['models'], data['training_results']
    except Exception as e:
        logger.error("Loading multiple models error: %s", e)
        raise ValueError(f"Failed to load multiple models from {filepath}: {e}")
