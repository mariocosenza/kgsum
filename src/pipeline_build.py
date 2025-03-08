import os
import pickle
from collections import Counter
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB


class ClassifierType(Enum):
    SVM = auto()
    NAIVE_BAYES = auto()


def majority_vote(predictions: List[Any]) -> Optional[Any]:
    if not predictions:
        return None
    vote_counts = Counter(predictions)
    return vote_counts.most_common(1)[0][0]


def _predict_category_for_instance(models: Dict[str, "KnowledgeGraphClassifier"],
                                   instance: Dict[str, Any]) -> Optional[Any]:
    predictions = []
    for feature, model in models.items():
        if feature not in instance:
            continue
        value = instance[feature]
        # Skip if value is None, an empty string, or NaN.
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue
        if not isinstance(value, list):
            # If the value is a scalar, check for NaN.
            if pd.isna(value):
                continue
            feature_data = [value]
        else:
            # For lists, skip if the list is empty.
            if len(value) == 0:
                continue
            feature_data = value
        try:
            pred = model.predict(feature_data)[0]
            predictions.append(pred)
        except Exception as e:
            print(f"Prediction error for feature '{feature}': {e}")
            continue
    return majority_vote(predictions)


def predict_category_multi(models: Dict[str, "KnowledgeGraphClassifier"],
                           instance: Union[Dict[str, Any], pd.DataFrame]) -> Union[Optional[Any], List[Optional[Any]]]:
    """
    Given a dictionary of trained models and an instance, predict the class using only available features.
    If a DataFrame is provided, predictions are made row-by-row and a list of predictions is returned.
    """
    if isinstance(instance, pd.DataFrame):
        return instance.apply(lambda row: _predict_category_for_instance(models, row.to_dict()), axis=1).tolist()
    else:
        return _predict_category_for_instance(models, instance)


def remove_empty_rows(frame: pd.DataFrame, labels: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(labels, str):
        labels = [labels]
    result = frame.copy()
    for label in labels:
        result = result.dropna(subset=[label])
        result = result[result[label] != '']
    return result


def train_multiple_models(training_data: pd.DataFrame,
                          feature_columns: List[str],
                          target_label: str = 'category',
                          classifier_type: ClassifierType = ClassifierType.SVM
                          ) -> Tuple[Dict[str, "KnowledgeGraphClassifier"], Dict[str, Any]]:
    models: Dict[str, KnowledgeGraphClassifier] = {}
    training_results: Dict[str, Any] = {}

    for feature in feature_columns:
        # Remove rows where the feature OR the target label are empty.
        df_feature = remove_empty_rows(training_data, [feature, target_label])
        if df_feature.empty:
            print(f"Skipping feature '{feature}': no data available.")
            continue

        # Check that there are at least two unique classes in the target.
        if df_feature[target_label].nunique() < 2:
            print(f"Skipping feature '{feature}': target label '{target_label}' has only one unique class: {df_feature[target_label].unique()}")
            continue

        print(f"Training model for feature '{feature}' with {len(df_feature)} examples.")
        model = KnowledgeGraphClassifier(classifier_type=classifier_type)
        try:
            result = model.train(df_feature, feature, target_label=target_label)
        except ValueError as ve:
            print(f"Skipping training for feature '{feature}': {ve}")
            continue
        models[feature] = model
        training_results[feature] = result

    return models, training_results


class KnowledgeGraphClassifier:
    def __init__(self, classifier_type: ClassifierType = ClassifierType.SVM):
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(max_features=10000, lowercase=True, ngram_range=(1, 2))
        self.model = None

    def _get_param_grid(self) -> Dict[str, List[Any]]:
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

    def _get_base_estimator(self):
        if self.classifier_type == ClassifierType.SVM:
            return svm.SVC(probability=True)
        else:
            return MultinomialNB()

    def _prepare_features(self, frame: pd.DataFrame,
                          feature_labels: Union[str, List[str]]) -> pd.Series:
        if isinstance(feature_labels, str):
            return frame[feature_labels].apply(lambda x: str(x) if not isinstance(x, str) else x)
        # Combine multiple feature columns into a single text column.
        combined_features = frame[feature_labels[0]].apply(lambda x: str(x) if not isinstance(x, str) else x).copy()
        for feature in feature_labels[1:]:
            combined_features = combined_features + " " + frame[feature].apply(
                lambda x: str(x) if not isinstance(x, str) else x)
        return combined_features

    def train(self, frame: pd.DataFrame,
              feature_labels: Union[str, List[str]],
              target_label: str = 'category',
              cv_folds: int = 2) -> Dict[str, Any]:
        frame = frame.reset_index(drop=True)
        # Drop rows with empty target values.
        frame = frame.dropna(subset=[target_label])
        frame = frame[frame[target_label] != '']
        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]

        # Check if there are at least two unique classes.
        if data_y.nunique() < 2:
            raise ValueError(f"Training data has only one unique class: {data_y.unique()}. At least two classes are required.")

        kf = KFold(n_splits=cv_folds, random_state=1, shuffle=True)
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

    def predict(self, data: Union[str, List[str], pd.Series, pd.DataFrame],
                feature_labels: Optional[Union[str, List[str]]] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if isinstance(data, pd.DataFrame) and feature_labels is not None:
            processed_data = self._prepare_features(data, feature_labels)
        elif isinstance(data, (pd.Series, list, str)):
            processed_data = data
        else:
            raise ValueError("Invalid data format. Provide DataFrame with feature_labels, Series, list, or string.")
        if isinstance(processed_data, str):
            processed_data = [processed_data]
        x_vectorized = self.vectorizer.transform(processed_data)
        return self.model.predict(x_vectorized)

    def predict_proba(self, data: Union[str, List[str], pd.Series, pd.DataFrame],
                      feature_labels: Optional[Union[str, List[str]]] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if isinstance(data, pd.DataFrame) and feature_labels is not None:
            processed_data = self._prepare_features(data, feature_labels)
        elif isinstance(data, (pd.Series, list, str)):
            processed_data = data
        else:
            raise ValueError("Invalid data format. Provide DataFrame with feature_labels, Series, list, or string.")
        if isinstance(processed_data, str):
            processed_data = [processed_data]
        x_vectorized = self.vectorizer.transform(processed_data)
        return self.model.predict_proba(x_vectorized)

    def save(self, filepath: Optional[str] = None):
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
    def load(cls, filepath: Optional[str] = None,
             classifier_type: ClassifierType = ClassifierType.SVM) -> "KnowledgeGraphClassifier":
        if filepath is None:
            filepath = f'../data/trained/model-{classifier_type.name}.pkl'
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(classifier_type=data['classifier_type'])
        instance.model = data['model']
        instance.vectorizer = data['vectorizer']
        return instance
