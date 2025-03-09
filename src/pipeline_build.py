from __future__ import annotations
import os
import pickle
import logging
from collections import Counter
from enum import Enum, auto
from typing import Any, BinaryIO, Tuple

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

# Suppress invalid cast warnings (if benign)
np.seterr(invalid='ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassifierType(Enum):
    SVM = auto()
    NAIVE_BAYES = auto()


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
            logger.info(f"Skipping '{feature}': target label has only one unique class: {df_feature[target_label].unique()}")
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


class KnowledgeGraphClassifier:
    def __init__(self, classifier_type: ClassifierType = ClassifierType.SVM):
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(max_features=10000, lowercase=True, ngram_range=(1, 2))
        self.model: GridSearchCV | None = None

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
                "C": [0,5, 1.0, 1,5, 2,0],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto", 0.01],
                "degree": [2, 3, 4, 5],
                "coef0": [0.0, 0.1],
                "tol": [1e-3, 1e-4],
                "class_weight": ["balanced", None],
                "decision_function_shape": ["ovr", "ovo"],
            }
        return {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "fit_prior": [True, False],
            "class_prior": [None, [0.3, 0.7], [0.5, 0.5]],
        }

    def _get_base_estimator(self):
        return svm.SVC(probability=True) if self.classifier_type == ClassifierType.SVM else MultinomialNB()

    def _get_cv_strategy(self, data_y: pd.Series, cv_folds: int) -> object:
        min_count = data_y.value_counts().min()
        if min_count < cv_folds:
            if min_count > 1:
                cv_folds = min_count
                logger.info(f"Adjusted cv_folds to {cv_folds} due to small class sizes.")
                return StratifiedKFold(n_splits=cv_folds, random_state=1, shuffle=True)
            else:
                logger.info("Using LeaveOneOut due to a class with a single sample.")
                return LeaveOneOut()
        return StratifiedKFold(n_splits=cv_folds, random_state=1, shuffle=True)

    def _process_input(self, data: Any, feature_labels: str | list[str] | None) -> Any:
        if feature_labels is not None:
            if isinstance(data, pd.DataFrame):
                return self._prepare_features(data, feature_labels)
            elif isinstance(data, pd.Series):
                return self._prepare_features(data.to_frame().T, feature_labels)
        return data

    def _vectorize(self, data: Any, feature_labels: str | list[str] | None) -> Any:
        processed = self._process_input(data, feature_labels)
        if isinstance(processed, str):
            processed = [processed]
        return self.vectorizer.transform(processed)

    def train(
        self,
        frame: pd.DataFrame,
        feature_labels: str | list[str],
        target_label: str = 'category',
        cv_folds: int = 2,
    ) -> dict[str, Any]:
        frame = frame.reset_index(drop=True)
        frame = frame.dropna(subset=[target_label])
        frame = frame[frame[target_label] != '']
        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]
        if data_y.nunique() < 2:
            raise ValueError(f"Only one unique class: {data_y.unique()}")

        cv_strategy = self._get_cv_strategy(data_y, cv_folds)
        grid = GridSearchCV(
            estimator=self._get_base_estimator(),
            param_grid=self._get_param_grid(),
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
        cv_scores = cross_val_score(grid.best_estimator_, x_vectorized, data_y, cv=cv_strategy, scoring='f1_weighted')
        mean_f1 = np.mean(cv_scores)
        logger.info(f"Training completed. Mean CV F1 Score: {mean_f1:.4f}")
        return {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_,
            'cv_scores': cv_scores,
            'cv_mean': mean_f1,
            'cv_std': np.std(cv_scores),
            'feature_labels': feature_labels,
            'classifier_type': self.classifier_type,
        }

    def predict(
        self,
        data: str | list[str] | pd.Series | pd.DataFrame,
        feature_labels: str | list[str] | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        x_vectorized = self._vectorize(data, feature_labels)
        return self.model.predict(x_vectorized)

    def predict_proba(
        self,
        data: str | list[str] | pd.Series | pd.DataFrame,
        feature_labels: str | list[str] | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        x_vectorized = self._vectorize(data, feature_labels)
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
        return instance


def save_multiple_models(
    models: dict[str, KnowledgeGraphClassifier],
    training_results: dict[str, Any],
    filepath: str | None = None,
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
