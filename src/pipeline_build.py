from __future__ import annotations

import logging
import os
import pickle
from collections import Counter
from enum import Enum, auto
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    LeaveOneOut,
)
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# Suppress invalid cast warnings (if benign)
np.seterr(invalid='ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassifierType(Enum):
    SVM = auto()
    NAIVE_BAYES = auto()
    CNN = auto()  # Added CNN option


def majority_vote(predictions: list[Any]) -> Any | None:
    return None if not predictions else Counter(predictions).most_common(1)[0][0]


def _predict_category_for_instance(
        models: dict[str, KnowledgeGraphClassifier],
        instance: dict[str, Any]
) -> Any | None:
    votes = []
    for feature, model in models.items():
        if feature not in instance:
            continue
        value = instance[feature]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        # Ensure value is a list
        feature_data = value if isinstance(value, list) else [value]
        if not feature_data:
            continue
        try:
            pred = model.predict(feature_data)[0]
            votes.append(pred)
        except Exception as err:
            logger.error(f"Prediction error for '{feature}': {err}")
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


# -------------------------------
# Train Multiple Models
# -------------------------------
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
            logger.info(f"Skipping '{feature}': no data available.")
            continue
        if df_feature[target_label].nunique() < 2:
            logger.info(
                f"Skipping '{feature}': target label has only one unique class: {df_feature[target_label].unique()}")
            continue
        logger.info(f"Training model for '{feature}' with {len(df_feature)} examples.")
        model = KnowledgeGraphClassifier(classifier_type=classifier_type)
        try:
            result = model.train(df_feature, feature, target_label=target_label)
        except ValueError as err:
            logger.info(f"Skipping '{feature}': {err}")
            continue
        models[feature] = model
        training_results[feature] = result
    return models, training_results


# -------------------------------
# KnowledgeGraphClassifier Class
# -------------------------------
class KnowledgeGraphClassifier:
    def __init__(self, classifier_type: ClassifierType = ClassifierType.SVM):
        self.classifier_type = classifier_type
        # For SVM and Naive Bayes, use TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=100000, lowercase=True, ngram_range=(1, 2))
        self.model: GridSearchCV | Sequential | None = None
        # Attributes used for the CNN branch
        self.tokenizer: Tokenizer | None = None
        self.max_length: int | None = None
        # Store vocabulary size for CNN
        self.vocab_size: int | None = None

    @staticmethod
    def _prepare_features(frame: pd.DataFrame, feature_labels: str | list[str]) -> pd.Series:
        if isinstance(feature_labels, str):
            return frame[feature_labels].astype(str)
        combined = frame[feature_labels[0]].astype(str)
        for label in feature_labels[1:]:
            combined += " " + frame[label].astype(str)
        return combined

    def _get_param_grid(self) -> dict[str, list[Any]]:
        if self.classifier_type == ClassifierType.SVM:
            return {
                "C": [0.5, 1.0, 1.5, 2.0],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto", 0.01],
                "degree": [2, 3, 5],
                "coef0": [0.0, 0.1],
                "tol": [1e-3, 1e-4],
                "class_weight": ["balanced", None],
                "decision_function_shape": ["ovr", "ovo"],
            }
        elif self.classifier_type == ClassifierType.NAIVE_BAYES:
            # Note: We'll remove "class_prior" if the number of classes != 2 later.
            return {
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "fit_prior": [True, False],
                "class_prior": [None, [0.3, 0.7], [0.5, 0.5]],
            }
        return {}

    def _get_base_estimator(self):
        if self.classifier_type == ClassifierType.SVM:
            return svm.SVC(probability=True)
        elif self.classifier_type == ClassifierType.NAIVE_BAYES:
            return MultinomialNB()
        elif self.classifier_type == ClassifierType.CNN:
            return None

    @staticmethod
    def _get_cv_strategy(data_y: pd.Series, cv_folds: int) -> object:
        min_count = data_y.value_counts().min()
        if min_count < cv_folds:
            if min_count > 1:
                cv_folds = min_count
                logger.info(f"Adjusted cv_folds to {cv_folds} due to small class sizes.")
                return StratifiedKFold(n_splits=cv_folds, random_state=42, shuffle=True)
            else:
                logger.info("Using LeaveOneOut due to a class with a single sample.")
                return LeaveOneOut()
        return StratifiedKFold(n_splits=cv_folds, random_state=42, shuffle=True)

    def _process_input(self, data: Any, feature_labels: str | list[str] | None) -> Any:
        if feature_labels is not None:
            if isinstance(data, pd.DataFrame):
                return self._prepare_features(data, feature_labels)
            elif isinstance(data, pd.Series):
                return self._prepare_features(data.to_frame().T, feature_labels)
        return data

    def _vectorize(self, data: Any, feature_labels: str | list[str] | None) -> Any:
        processed = self._process_input(data, feature_labels)
        if self.classifier_type == ClassifierType.CNN:
            texts = [processed] if isinstance(processed, str) else processed
            sequences = self.tokenizer.texts_to_sequences(texts)
            return pad_sequences(sequences, maxlen=self.max_length)
        else:
            if isinstance(processed, str):
                processed = [processed]
            return self.vectorizer.transform(processed)

    def train(
            self,
            frame: pd.DataFrame,
            feature_labels: str | list[str],
            target_label: str = 'category',
            cv_folds: int = 5,
    ) -> dict[str, Any]:
        frame = frame.reset_index(drop=True)
        frame = frame.dropna(subset=[target_label])
        frame = frame[frame[target_label] != '']
        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]
        if data_y.nunique() < 2:
            raise ValueError(f"Only one unique class: {data_y.unique()}")

        if self.classifier_type == ClassifierType.CNN:
            # Initialize tokenizer without limiting num_words to ensure all tokens are captured
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(data_x)
            sequences = self.tokenizer.texts_to_sequences(data_x)
            self.max_length = max(len(seq) for seq in sequences)
            x_data = pad_sequences(sequences, maxlen=self.max_length)

            # Calculate vocabulary size with a buffer margin
            self.vocab_size = len(self.tokenizer.word_index) + 1  # +1 for the padding token (0)
            logger.info(f"Vocabulary size: {self.vocab_size}")

            num_classes = data_y.nunique()
            if num_classes > 2:
                labels_encoded = data_y.factorize()[0]
                y_data = to_categorical(labels_encoded, num_classes=num_classes)
            else:
                unique = list(data_y.unique())
                mapping = {unique[0]: 0, unique[1]: 1}
                y_data = data_y.map(mapping).values

            model = Sequential()
            # Use the calculated vocab_size instead of fixed 10000
            model.add(Embedding(input_dim=self.vocab_size, output_dim=128))
            model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
            model.add(GlobalMaxPooling1D())
            model.add(Dense(64, activation='relu'))
            if num_classes > 2:
                model.add(Dense(num_classes, activation='softmax'))
                loss = 'categorical_crossentropy'
            else:
                model.add(Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
            history = model.fit(x_data, y_data, epochs=10, batch_size=32, verbose=1)
            self.model = model
            logger.info("CNN training completed.")
            return {
                'history': history.history,
                'epochs': 10,
                'num_classes': num_classes,
                'max_length': self.max_length,
                'vocab_size': self.vocab_size,
            }
        else:
            param_grid = self._get_param_grid()
            if self.classifier_type == ClassifierType.NAIVE_BAYES:
                # Remove class_prior if number of classes is not 2.
                if data_y.nunique() != 2 and 'class_prior' in param_grid:
                    del param_grid['class_prior']
            cv_strategy = self._get_cv_strategy(data_y, cv_folds)
            grid = GridSearchCV(
                estimator=self._get_base_estimator(),
                param_grid=param_grid,
                cv=cv_strategy,
                scoring='f1_weighted',
                return_train_score=True,
                verbose=1,
                n_jobs=-1,
                error_score=np.nan,
            )
            x_vectorized = self.vectorizer.fit_transform(data_x)
            grid.fit(x_vectorized, data_y)
            self.model = grid
            cv_scores = cross_val_score(grid.best_estimator_, x_vectorized, data_y, cv=cv_strategy,
                                        scoring='f1_weighted')
            mean_f1 = np.mean(cv_scores)
            logger.info(f"Training completed. Mean CV F1 Score: {mean_f1:.4f}")
            return {
                'best_params': grid.best_params_,
                'best_score': grid.best_score_,
                'cv_scores': cv_scores,
                'cv_mean': mean_f1,
                'cv_std': np.std(cv_scores),
                'feature_labels': feature_labels,
                'classifier_type': self.classifier_type.name,
            }

    def predict(
            self,
            data: str | list[str] | pd.Series | pd.DataFrame,
            feature_labels: str | list[str] | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        x_vectorized = self._vectorize(data, feature_labels)
        if self.classifier_type == ClassifierType.CNN:
            preds = self.model.predict(x_vectorized)
            if preds.shape[-1] > 1:
                return np.argmax(preds, axis=1)
            else:
                return (preds > 0.5).astype(int).flatten()
        else:
            return self.model.predict(x_vectorized)

    def predict_proba(
            self,
            data: str | list[str] | pd.Series | pd.DataFrame,
            feature_labels: str | list[str] | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        x_vectorized = self._vectorize(data, feature_labels)
        if self.classifier_type == ClassifierType.CNN:
            return self.model.predict(x_vectorized)
        else:
            return self.model.predict_proba(x_vectorized)

    def save(self, filepath: str | None = None):
        if self.model is None:
            raise ValueError("No trained model to save.")
        if filepath is None:
            os.makedirs('../data/trained', exist_ok=True)
            filepath = f'../data/trained/model-{self.classifier_type.name}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'classifier_type': self.classifier_type,
                'tokenizer': self.tokenizer,
                'max_length': self.max_length,
                'vocab_size': self.vocab_size,
            }, f)

    @classmethod
    def load(cls, filepath: str | None = None,
             classifier_type: ClassifierType = ClassifierType.SVM) -> KnowledgeGraphClassifier:
        if filepath is None:
            filepath = f'../data/trained/model-{classifier_type.name}.pkl'
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(classifier_type=data['classifier_type'])
        instance.model = data['model']
        instance.vectorizer = data['vectorizer']
        # Load CNN-specific attributes if available
        if 'tokenizer' in data:
            instance.tokenizer = data['tokenizer']
        if 'max_length' in data:
            instance.max_length = data['max_length']
        if 'vocab_size' in data:
            instance.vocab_size = data['vocab_size']
        return instance


def save_multiple_models(
        models: dict[str, KnowledgeGraphClassifier],
        training_results: dict[str, Any],
        filepath: str | None = None
) -> None:
    if filepath is None:
        os.makedirs('../data/trained', exist_ok=True)
        filepath = '../data/trained/multiple_models.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump({'models': models, 'training_results': training_results}, f)
    logger.info(f"Multiple models saved to {filepath}")


def load_multiple_models(filepath: str | None = None) -> Tuple[dict[str, KnowledgeGraphClassifier], dict[str, Any]]:
    if filepath is None:
        filepath = '../data/trained/multiple_models.pkl'
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Multiple models loaded from {filepath}")
    return data['models'], data['training_results']