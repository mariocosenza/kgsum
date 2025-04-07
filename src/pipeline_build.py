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


from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    logging as hf_logging,
    get_linear_schedule_with_warmup,
)
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm

# New type aliases using Python 3.12 features
FeatureLabels: TypeAlias = str | list[str]
TextData = NewType('TextData', str)

hf_logging.set_verbosity_error()
np.seterr(invalid='ignore')

# Suppress warning regarding token_pattern when using custom tokenizer
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
        # Safe lookup: if the feature is absent, skip it.
        value = instance.get(feature, None)
        if value is None:
            logger.debug("Feature '%s' not found in instance; skipping.", feature)
            continue
        # Handle string values.
        if isinstance(value, str):
            if not value.strip():
                continue
            feature_data = [value]
        # Handle iterable types (list, set, tuple) that are not strings or bytes.
        elif isinstance(value, (list, set, tuple)) and not isinstance(value, (str, bytes)):
            feature_data = []
            for item in value:
                if item is None:
                    continue
                if not isinstance(item, str):
                    item = str(item)
                if not item.strip():
                    continue
                feature_data.append(item)
            if not feature_data:
                continue
        else:
            # For any other type, convert to string.
            if not isinstance(value, str):
                value = str(value)
            if not value.strip():
                continue
            feature_data = [value]
        try:
            pred = model.predict(feature_data)[0]
            f1 = getattr(model, "accuracy", 0.5)
            if pred is not None:
                votes.append((pred, f1, feature))
        except Exception as err:
            logger.error("Prediction error for '%s': %s", feature, err)
    return majority_vote(votes)


def predict_category_multi(
    models: dict[str, KnowledgeGraphClassifier],
    instance: dict[str, Any] | pd.DataFrame
) -> Any | list[Any | None]:
    if isinstance(instance, pd.DataFrame):
        return instance.apply(
            lambda row: _predict_category_for_instance(models, row.to_dict()), axis=1
        ).tolist()
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
    max_count = counts.max()
    groups = []
    for label, group in df.groupby(target_label):
        target_count = min(max_count, int(len(group) * max_factor))
        if len(group) < target_count:
            group_over = group.sample(target_count, replace=True, random_state=42)
            groups.append(group_over)
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
            logger.info("Skipping '%s': target label has only one unique class: %s", feature, df_feature[target_label].unique())
            continue
        counts = df_feature[target_label].value_counts()
        if (counts.min() / counts.max()) < 0.75:
            logger.info("Feature '%s' is unbalanced. Applying limited oversampling.", feature)
            df_feature = oversample_dataframe(df_feature, target_label, max_factor=2.0)
        logger.info("Training model for '%s' with %s examples.", feature, len(df_feature))
        model = KnowledgeGraphClassifier(classifier_type=classifier_type, balance_classes=True)
        try:
            result = model.train(df_feature, feature, target_label=target_label)
        except ValueError as err:
            logger.info("Skipping '%s': %s", feature, err)
            continue
        except Exception as err:
            logger.error("Unexpected error training model for '%s': %s", feature, err)
            continue
        models[feature] = model
        training_results[feature] = result

    return models, training_results


class KnowledgeGraphClassifier:
    def __init__(
        self, classifier_type: ClassifierType = ClassifierType.SVM, balance_classes: bool = True
    ) -> None:
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
        self.accuracy: float = 0.5  # Default performance weight

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\[\s*\]', '', text)
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\{\s*\}', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def custom_tokenizer(self, text: str) -> list[str]:
        return self._clean_text(text).split()

    def _prepare_features(self, frame: pd.DataFrame, feature_labels: FeatureLabels) -> pd.Series:
        if isinstance(feature_labels, str):
            features = frame[feature_labels].fillna('').astype(str)
        else:
            combined = frame[feature_labels[0]].fillna('').astype(str)
            for label in feature_labels[1:]:
                combined += " " + frame[label].fillna('').astype(str)
            features = combined
        return features.apply(self._clean_text)

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
            return {"alpha": [0.5, 1, 2, 5, 10], "fit_prior": [True, False]}
        elif self.classifier_type == ClassifierType.KNN:
            return {
                "n_neighbors": [11, 15, 21, 25, 31],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
                "leaf_size": [20, 30, 40]
            }
        return {}

    def _get_base_estimator(self):
        if self.classifier_type == ClassifierType.SVM:
            return svm.SVC(probability=True, random_state=67)
        elif self.classifier_type == ClassifierType.NAIVE_BAYES:
            return MultinomialNB()
        elif self.classifier_type == ClassifierType.KNN:
            return KNeighborsClassifier()
        else:
            raise ValueError(f"Base estimator not defined for classifier type: {self.classifier_type}")

    def _get_cv_strategy(self, data_y: pd.Series, cv_folds: int) -> StratifiedKFold:
        return StratifiedKFold(n_splits=2, random_state=67, shuffle=True)

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
            return self.vectorizer.transform(processed)

    def train(
        self,
        frame: pd.DataFrame,
        feature_labels: FeatureLabels,
        target_label: str = 'category',
        cv_folds: int = 5,
        max_length: int = 512
    ) -> dict[str, Any]:
        self.feature_labels = feature_labels
        frame = frame.reset_index(drop=True)
        frame = frame.dropna(subset=[target_label])
        frame = frame[frame[target_label] != '']
        # Check for imbalance and apply limited oversampling if necessary
        if self.balance_classes:
            counts = frame[target_label].value_counts()
            if (counts.min() / counts.max()) < 0.75:
                logger.info("Classes are unbalanced. Applying limited oversampling to rebalance.")
                frame = oversample_dataframe(frame, target_label, max_factor=2.0)
        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]
        unique_labels = sorted(data_y.unique())
        self.target_label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.target_label_inverse_mapping = {idx: label for label, idx in self.target_label_mapping.items()}
        if data_y.nunique() < 2:
            raise ValueError(f"Only one unique class: {data_y.unique()}")

        if self.classifier_type == ClassifierType.ROBERTA:
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.max_length = max_length  # Set max_length as provided
            texts = data_x.tolist()
            texts = [self._clean_text(text) for text in texts]
            encoding = self.roberta_tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            num_classes = data_y.nunique()
            y_encoded = data_y.map(self.target_label_mapping).values
            labels_tensor = torch.tensor(y_encoded, dtype=torch.long)

            dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
            val_size = int(0.1 * len(dataset))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16)

            self.model = RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                num_labels=num_classes
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

            if self.balance_classes:
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_encoded)
                class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_encoded)
                class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
            else:
                class_weights_tensor = None

            optimizer = AdamW(self.model.parameters(), lr=2e-5)
            num_epochs = 10
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
            )

            best_val_loss = float('inf')
            patience = 3
            epochs_without_improvement = 0
            best_model_state = None

            avg_train_loss = 0.0

            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0.0
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
                    batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]
                    optimizer.zero_grad()
                    if not isinstance(self.model, RobertaForSequenceClassification):
                        raise ValueError("Expected a RobertaForSequenceClassification model, but got a different type.")
                    outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                    logits = outputs.logits
                    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor) if class_weights_tensor is not None else torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.detach().item()
                avg_train_loss = total_loss / len(train_loader)

                self.model.eval()
                total_val_loss = 0.0
                all_preds: list[Any] = []
                all_labels: list[Any] = []
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                    batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]
                    outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                    logits = outputs.logits
                    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor) if class_weights_tensor is not None else torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits, batch_labels)
                    total_val_loss += loss.detach().item()
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                avg_val_loss = total_val_loss / (len(val_loader) if len(val_loader) > 0 else 1)
                val_f1 = f1_score(all_labels, all_preds, average='weighted')
                logger.info("Epoch %s - Train Loss: %.4f, Val Loss: %.4f, Val F1: %.4f", epoch+1, avg_train_loss, avg_val_loss, val_f1)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        logger.info("Early stopping triggered.")
                        break

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids.to(device), attention_mask=attention_mask.to(device))
                logits = outputs.logits
            if num_classes > 2:
                final_preds = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                final_preds = (probs > 0.5).astype(int)
            final_f1 = f1_score(y_encoded, final_preds, average='weighted')
            final_acc = accuracy_score(y_encoded, final_preds)
            self.accuracy = final_f1  # Store training F1 score for voting
            conf_mat = confusion_matrix(y_encoded, final_preds)
            legend = "Legend: Rows=True Labels, Columns=Predicted Labels"
            logger.info("Final RoBERTa F1 Score: %.4f", final_f1)
            logger.info("Final RoBERTa Accuracy Score: %.4f", final_acc)
            logger.info("Confusion Matrix:\n%s\n%s", conf_mat, legend)
            training_history = {"train_loss": avg_train_loss, "val_loss": best_val_loss}
            return {
                'history': training_history,
                'final_f1': final_f1,
                'final_accuracy': final_acc,
                'confusion_matrix': conf_mat,
                'confusion_matrix_legend': legend,
                'num_classes': num_classes,
                'max_length': self.max_length,
                'target_label_mapping': self.target_label_mapping,
                'feature_labels': feature_labels,
                'classifier_type': self.classifier_type.name,
            }
        else:
            param_grid = self._get_param_grid()
            cv_strategy = self._get_cv_strategy(data_y, cv_folds)
            try:
                x_vectorized = self.vectorizer.fit_transform(data_x)
                if x_vectorized.shape[1] == 0:
                    raise ValueError("Vectorization resulted in no features. Check input data.")
            except Exception as e:
                logger.error("Vectorization error: %s", e)
                raise ValueError(f"Failed to vectorize text data: {e}")
            y_encoded = data_y.map(self.target_label_mapping).values

            # Use a custom scorer with zero_division set to 0 to avoid scoring errors.
            scorer = make_scorer(f1_score, average='weighted', zero_division=0)
            try:
                grid = GridSearchCV(
                    estimator=self._get_base_estimator(),
                    param_grid=param_grid,
                    cv=cv_strategy,
                    scoring=scorer,
                    return_train_score=True,
                    verbose=1,
                    n_jobs=-1,
                    error_score=np.nan,
                )
                grid.fit(x_vectorized, y_encoded)
            except Exception as e:
                logger.error("Grid search error: %s", e)
                raise ValueError(f"Grid search failed: {e}")
            try:
                y_pred = grid.predict(x_vectorized)
                final_f1 = f1_score(y_encoded, y_pred, average='weighted')
                final_acc = accuracy_score(y_encoded, y_pred)
                self.accuracy = final_f1  # Store training F1 score for voting
                conf_mat = confusion_matrix(y_encoded, y_pred)
                legend = "Legend: Rows=True Labels, Columns=Predicted Labels"
                logger.info("Final Model F1 Score: %.4f", final_f1)
                logger.info("Final Model Accuracy Score: %.4f", final_acc)
                logger.info("Confusion Matrix:\n%s\n%s", conf_mat, legend)
            except Exception as e:
                logger.error("Error during final model evaluation: %s", e)
                final_f1 = np.nan
                final_acc = np.nan
                conf_mat = None
                legend = ""
            self.model = grid
            return {
                'best_params': grid.best_params_,
                'final_f1': final_f1,
                'final_accuracy': final_acc,
                'confusion_matrix': conf_mat,
                'confusion_matrix_legend': legend,
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
        if feature_labels is None and self.feature_labels is not None:
            feature_labels = self.feature_labels
        x_vectorized = self._vectorize(data, feature_labels)
        try:
            if self.classifier_type == ClassifierType.ROBERTA:
                self.model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x_vectorized = {k: v.to(device) for k, v in x_vectorized.items()}
                with torch.no_grad():
                    outputs = self.model(**x_vectorized)
                logits = outputs.logits
                num_labels = self.model.config.num_labels
                if num_labels > 1:
                    numerical_preds = torch.argmax(logits, dim=1).cpu().numpy()
                else:
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    numerical_preds = (probs > 0.5).astype(int)
            else:
                if hasattr(self.model, "predict"):
                    numerical_preds = self.model.predict(x_vectorized)
                else:
                    raise ValueError("The model does not support prediction via predict().")
            if self.target_label_inverse_mapping:
                return np.array([self.target_label_inverse_mapping[int(idx)] for idx in numerical_preds])
            else:
                return numerical_preds
        except Exception as e:
            logger.error("Prediction error: %s", e)
            raise ValueError(f"Failed to make predictions: {e}")

    def predict_proba(
        self,
        data: str | list[str] | pd.Series | pd.DataFrame,
        feature_labels: FeatureLabels | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if feature_labels is None and self.feature_labels is not None:
            feature_labels = self.feature_labels
        x_vectorized = self._vectorize(data, feature_labels)
        try:
            if self.classifier_type == ClassifierType.ROBERTA:
                self.model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x_vectorized = {k: v.to(device) for k, v in x_vectorized.items()}
                with torch.no_grad():
                    outputs = self.model(**x_vectorized)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                return probs
            else:
                if hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(x_vectorized)
                else:
                    raise ValueError("The model does not support probability predictions via predict_proba().")
        except Exception as e:
            logger.error("Error in predict_proba: %s", e)
            raise ValueError(f"Failed to get probability estimates: {e}")

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
            logger.error("Error saving model: %s", e)
            raise ValueError(f"Failed to save model: {e}")

    @classmethod
    def load(cls, filepath: str | None = None, classifier_type: ClassifierType = ClassifierType.SVM) -> Self:
        if filepath is None:
            filepath = f'../data/trained/model-{classifier_type.name}.pkl'
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            classifier_from_file = data.get('classifier_type')
            if not isinstance(classifier_from_file, ClassifierType):
                classifier_from_file = classifier_type
            instance = cls(classifier_type=classifier_from_file)
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
            logger.error("Error loading model: %s", e)
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
        logger.error("Error saving multiple models: %s", e)
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
        logger.error("Error loading multiple models: %s", e)
        raise ValueError(f"Failed to load multiple models from {filepath}: {e}")
