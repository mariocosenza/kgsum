import os
import pickle
from enum import Enum, auto
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

class ClassifierType(Enum):
    SVM = auto()
    NAIVE_BAYES = auto()


class KnowledgeGraphClassifier:
    def __init__(self, classifier_type: ClassifierType = ClassifierType.SVM):
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(max_features=10000, lowercase=True, ngram_range=(1, 2))
        self.model = None

    def _get_param_grid(self: "KnowledgeGraphClassifier") -> Dict[str, List[Any]]:
        if self.classifier_type == ClassifierType.SVM:
            return {
                "C": [0.1, 0.5, 1.0, 1.5, 2.0, 5.0],
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "degree": [2, 3, 4, 5],
                "gamma": ["scale", "auto", 0.1, 0.01, 0.001],
                "class_weight": ["balanced", None]
            }
        else:
            return {
                "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
                "fit_prior": [True, False],
                "class_prior": [None, [0.3, 0.7], [0.5, 0.5]],
            }

    def _get_base_estimator(self: "KnowledgeGraphClassifier"):
        if self.classifier_type == ClassifierType.SVM:
            return svm.SVC(probability=True)
        else:
            return MultinomialNB()

    def _prepare_features(self: "KnowledgeGraphClassifier", frame: pd.DataFrame,
                          feature_labels: str | List[str]) -> pd.Series:
        if isinstance(feature_labels, str):
            return frame[feature_labels].apply(lambda x: str(x) if not isinstance(x, str) else x)

        # Convert all columns to strings first
        combined_features = frame[feature_labels[0]].apply(lambda x: str(x) if not isinstance(x, str) else x).copy()
        for feature in feature_labels[1:]:
            combined_features = combined_features + " " + frame[feature].apply(
                lambda x: str(x) if not isinstance(x, str) else x)

        return combined_features

    def train(self: "KnowledgeGraphClassifier",
              frame: pd.DataFrame,
              feature_labels: str | List[str],
              target_label: str = 'category',
              cv_folds: int = 5) -> Dict[str, Any]:
        frame = frame.reset_index(drop=True)

        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]

        kf = KFold(n_splits=cv_folds, random_state=42, shuffle=True)

        grid = GridSearchCV(
            estimator=self._get_base_estimator(),
            param_grid=self._get_param_grid(),
            cv=kf,
            scoring='f1_weighted',
            return_train_score=True,
            verbose=1,
            n_jobs=-1
        )

        x_vectorized = self.vectorizer.fit_transform(data_x)

        grid.fit(x_vectorized, data_y)

        self.model = grid

        cv_scores = cross_val_score(
            grid.best_estimator_,
            x_vectorized,
            data_y,
            cv=kf,
            scoring='f1_weighted'
        )

        results = {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'feature_labels': feature_labels,
            'classifier_type': self.classifier_type
        }

        return results

    def predict(self: "KnowledgeGraphClassifier", data: str | List[str] | pd.Series | pd.DataFrame,
                feature_labels: Optional[str | List[str]] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(data, pd.DataFrame) and feature_labels is not None:
            processed_data = self._prepare_features(data, feature_labels)
        elif isinstance(data, pd.Series) or isinstance(data, list) or isinstance(data, str):
            processed_data = data
        else:
            raise ValueError("Invalid data format. Provide DataFrame with feature_labels, Series, list, or string.")

        if isinstance(processed_data, str):
            processed_data = [processed_data]

        x_vectorized = self.vectorizer.transform(processed_data)

        return self.model.predict(x_vectorized)

    def predict_proba(self: "KnowledgeGraphClassifier", data: str | List[str] | pd.Series | pd.DataFrame,
                      feature_labels: Optional[str | List[str]] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(data, pd.DataFrame) and feature_labels is not None:
            processed_data = self._prepare_features(data, feature_labels)
        elif isinstance(data, pd.Series) or isinstance(data, list) or isinstance(data, str):
            processed_data = data
        else:
            raise ValueError("Invalid data format. Provide DataFrame with feature_labels, Series, list, or string.")

        if isinstance(processed_data, str):
            processed_data = [processed_data]

        x_vectorized = self.vectorizer.transform(processed_data)

        return self.model.predict_proba(x_vectorized)

    def save(self: "KnowledgeGraphClassifier", filepath: str = None):
        if self.model is None:
            raise ValueError("No trained model to save.")

        if filepath is None:
            os.makedirs('../data/trained', exist_ok=True)
            filepath = f'../data/trained/model-{self.classifier_type.name}.pkl'

        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'classifier_type': self.classifier_type
            }, f)

    @classmethod
    def load(cls, filepath: str = None,
             classifier_type: ClassifierType = ClassifierType.SVM) -> "KnowledgeGraphClassifier":
        if filepath is None:
            filepath = f'../data/trained/model-{classifier_type.name}.pkl'

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(classifier_type=data['classifier_type'])
        instance.model = data['model']
        instance.vectorizer = data['vectorizer']

        return instance


def remove_empty_rows(frame: pd.DataFrame, labels: List[str] | str) -> pd.DataFrame:
    if isinstance(labels, str):
        labels = [labels]

    result = frame.copy()

    for label in labels:
        result = result[result[label] != '']

    return result


