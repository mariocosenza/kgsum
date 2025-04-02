from __future__ import annotations

import logging
import os
import pickle
import re
from collections import Counter
from enum import Enum, auto
from typing import Any, Tuple, Union, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    LeaveOneOut,
    RepeatedStratifiedKFold,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score

# For BERT
from transformers import TFBertForSequenceClassification, BertTokenizer

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress invalid cast warnings (if benign)
np.seterr(invalid='ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassifierType(Enum):
    SVM = auto()
    NAIVE_BAYES = auto()
    BERT = auto()  # New option


def majority_vote(predictions: List[Any]) -> Any:
    """Return the most common prediction, or None if list is empty."""
    if not predictions:
        return None
    return Counter(predictions).most_common(1)[0][0]


def _predict_category_for_instance(
        models: Dict[str, 'KnowledgeGraphClassifier'],
        instance: Dict[str, Any]
) -> Optional[Any]:
    """Predict category for a single instance using multiple models."""
    votes = []
    for feature, model in models.items():
        if feature not in instance:
            continue
        value = instance[feature]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
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
        models: Dict[str, 'KnowledgeGraphClassifier'],
        instance: Union[Dict[str, Any], pd.DataFrame]
) -> Union[Any, List[Optional[Any]]]:
    """Predict categories for multiple instances or a single instance."""
    if isinstance(instance, pd.DataFrame):
        return instance.apply(lambda row: _predict_category_for_instance(models, row.to_dict()), axis=1).tolist()
    return _predict_category_for_instance(models, instance)


def remove_empty_rows(frame: pd.DataFrame, labels: Union[str, List[str]]) -> pd.DataFrame:
    """Remove rows with empty values in specified columns."""
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
        feature_columns: List[str],
        target_label: str = 'category',
        classifier_type: ClassifierType = ClassifierType.SVM
) -> Tuple[Dict[str, 'KnowledgeGraphClassifier'], Dict[str, Any]]:
    """Train multiple models, one for each feature column."""
    models: Dict[str, KnowledgeGraphClassifier] = {}
    training_results: Dict[str, Any] = {}

    for feature in feature_columns:
        df_feature = remove_empty_rows(training_data, [feature, target_label])
        if df_feature.empty:
            logger.info(f"Skipping '{feature}': no data available.")
            continue

        if df_feature[target_label].nunique() < 2:
            logger.info(
                f"Skipping '{feature}': target label has only one unique class: {df_feature[target_label].unique()}"
            )
            continue

        class_counts = df_feature[target_label].value_counts()
        if class_counts.min() < 3:
            logger.info(
                f"Warning: Feature '{feature}' has a class with only {class_counts.min()} samples. "
                "This might lead to poor model performance."
            )

        logger.info(f"Training model for '{feature}' with {len(df_feature)} examples.")
        model = KnowledgeGraphClassifier(classifier_type=classifier_type)
        try:
            result = model.train(df_feature, feature, target_label=target_label)
        except ValueError as err:
            logger.info(f"Skipping '{feature}': {err}")
            continue
        except Exception as err:
            logger.error(f"Unexpected error training model for '{feature}': {err}")
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
        # For classical models, use a custom tokenizer with TF-IDF.
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.custom_tokenizer,
            token_pattern=None,
            max_features=10000,
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
            use_idf=True,
            norm='l2',
            stop_words='english'
        )
        self.model: Optional[Union[GridSearchCV, tf.keras.Model, TFBertForSequenceClassification]] = None
        # For BERT, we will store the BERT tokenizer separately.
        self.bert_tokenizer: Optional[BertTokenizer] = None

        # Mapping for target labels
        self.target_label_mapping: Optional[Dict[Any, int]] = None
        self.target_label_inverse_mapping: Optional[Dict[int, Any]] = None
        # Stored feature_labels for future predictions
        self.feature_labels: Optional[Union[str, List[str]]] = None
        # For BERT, maximum sequence length
        self.max_length: Optional[int] = None

    def custom_tokenizer(self, text: str) -> List[str]:
        """
        Custom tokenizer to extract URLs as tokens and then tokenize remaining text.
        This preserves meaningful URI components.
        """
        url_regex = r'(https?://[^\s]+)'
        urls = re.findall(url_regex, text)
        text_without_urls = re.sub(url_regex, ' ', text)
        word_tokens = re.findall(r'\b\w+\b', text_without_urls)
        return urls + word_tokens

    def _clean_text(self, text: str) -> str:
        """Clean text by removing empty brackets and extra spaces."""
        text = re.sub(r'\[\s*\]', '', text)
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\{\s*\}', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _prepare_features(self, frame: pd.DataFrame, feature_labels: Union[str, List[str]]) -> pd.Series:
        """Prepare features from the DataFrame (single or multiple columns)."""
        if isinstance(feature_labels, str):
            features = frame[feature_labels].fillna('').astype(str)
        else:
            combined = frame[feature_labels[0]].fillna('').astype(str)
            for label in feature_labels[1:]:
                combined += " " + frame[label].fillna('').astype(str)
            features = combined
        return features.apply(self._clean_text)

    def _get_param_grid(self) -> Union[Dict[str, List[Any]], List[Dict[str, Any]]]:
        """Expanded parameter grid for classical models."""
        if self.classifier_type == ClassifierType.SVM:
            return [
                {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "kernel": ["linear"],
                    "class_weight": [None, "balanced"],
                },
                {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "kernel": ["rbf"],
                    "gamma": ["scale", "auto"],
                    "class_weight": [None, "balanced"],
                },
                {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "kernel": ["poly"],
                    "degree": [2, 3],
                    "gamma": ["scale", "auto"],
                    "class_weight": [None, "balanced"],
                },
            ]
        elif self.classifier_type == ClassifierType.NAIVE_BAYES:
            return {
                "alpha": [0.001, 0.01, 0.1, 1, 10],
                "fit_prior": [True, False],
            }
        return {}

    def _get_base_estimator(self):
        """Return the base estimator for classical models."""
        if self.classifier_type == ClassifierType.SVM:
            return svm.SVC(probability=True, random_state=42)
        elif self.classifier_type == ClassifierType.NAIVE_BAYES:
            return MultinomialNB()
        else:
            raise ValueError(f"Base estimator not defined for classifier type: {self.classifier_type}")

    def _get_cv_strategy(self, data_y: pd.Series, cv_folds: int) -> object:
        """
        Choose an appropriate cross-validation strategy.
        Use RepeatedStratifiedKFold with 3 repeats for larger datasets;
        fallback to LeaveOneOut for very small or imbalanced data.
        """
        min_count = data_y.value_counts().min()
        if len(data_y) < 10 or min_count < 3:
            return LeaveOneOut()
        elif len(data_y) >= 30:
            return RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=3, random_state=42)
        else:
            return StratifiedKFold(n_splits=cv_folds, random_state=42, shuffle=True)

    def _process_input(self, data: Any, feature_labels: Optional[Union[str, List[str]]]) -> Any:
        """Process input data for prediction."""
        if feature_labels is None:
            feature_labels = self.feature_labels
            if feature_labels is None:
                raise ValueError("No feature_labels provided or stored during training")
        if isinstance(data, pd.DataFrame):
            return self._prepare_features(data, feature_labels)
        elif isinstance(data, pd.Series):
            return self._prepare_features(data.to_frame().T, feature_labels)
        else:
            if isinstance(data, str):
                return self._clean_text(data)
            elif isinstance(data, list) and all(isinstance(item, str) for item in data):
                return [self._clean_text(item) for item in data]
            return data

    def _vectorize(self, data: Any, feature_labels: Optional[Union[str, List[str]]]) -> Any:
        """
        Vectorize input data for model prediction.
        For classical models, we use the TF-IDF vectorizer.
        For BERT, we use the BERT tokenizer.
        """
        processed = self._process_input(data, feature_labels)
        if self.classifier_type == ClassifierType.BERT:
            texts = [processed] if isinstance(processed, str) else processed
            encoding = self.bert_tokenizer.batch_encode_plus(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='tf'
            )
            return encoding
        else:
            if isinstance(processed, str):
                processed = [processed]
            return self.vectorizer.transform(processed)

    def train(
            self,
            frame: pd.DataFrame,
            feature_labels: Union[str, List[str]],
            target_label: str = 'category',
            cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train the classifier on the provided data."""
        self.feature_labels = feature_labels
        frame = frame.reset_index(drop=True)
        frame = frame.dropna(subset=[target_label])
        frame = frame[frame[target_label] != '']
        data_x = self._prepare_features(frame, feature_labels)
        data_y = frame[target_label]
        unique_labels = sorted(data_y.unique())
        self.target_label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.target_label_inverse_mapping = {idx: label for label, idx in self.target_label_mapping.items()}
        if data_y.nunique() < 2:
            raise ValueError(f"Only one unique class: {data_y.unique()}")

        if self.classifier_type == ClassifierType.BERT:
            # BERT branch
            # Initialize the BERT tokenizer and set maximum sequence length.
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.max_length = 128
            texts = data_x.tolist()
            encoding = self.bert_tokenizer.batch_encode_plus(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='tf'
            )
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            num_classes = data_y.nunique()
            y_encoded = data_y.map(self.target_label_mapping).values
            if num_classes > 2:
                y_data = to_categorical(y_encoded, num_classes=num_classes)
            else:
                y_data = y_encoded

            # Load a pre-trained BERT model for sequence classification
            self.model = TFBertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=num_classes,
                output_attentions=False,
                output_hidden_states=False,
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
            self.model.compile(optimizer=optimizer, loss=self.model.compute_loss, metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = self.model.fit(
                {'input_ids': input_ids, 'attention_mask': attention_mask},
                y_data,
                validation_split=0.1,
                epochs=5,
                batch_size=16,
                callbacks=[early_stopping],
                verbose=1
            )
            preds = self.model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})
            logits = preds.logits
            if num_classes > 2:
                final_preds = np.argmax(logits, axis=1)
            else:
                final_preds = (tf.sigmoid(logits) > 0.5).numpy().flatten().astype(int)
            final_f1 = f1_score(y_encoded, final_preds, average='weighted')
            final_acc = accuracy_score(y_encoded, final_preds)
            logger.info(f"Final BERT F1 Score: {final_f1:.4f}")
            logger.info(f"Final BERT Accuracy Score: {final_acc:.4f}")
            self.model.train_history = history.history
            return {
                'history': history.history,
                'final_f1': final_f1,
                'final_accuracy': final_acc,
                'num_classes': num_classes,
                'max_length': self.max_length,
                'target_label_mapping': self.target_label_mapping,
                'feature_labels': feature_labels,
                'classifier_type': self.classifier_type.name,
            }
        else:
            # Classical models branch (SVM or Naive Bayes)
            param_grid = self._get_param_grid()
            cv_strategy = self._get_cv_strategy(data_y, cv_folds)
            try:
                x_vectorized = self.vectorizer.fit_transform(data_x)
                if x_vectorized.shape[1] == 0:
                    raise ValueError("Vectorization resulted in no features. Check input data.")
            except Exception as e:
                logger.error(f"Vectorization error: {e}")
                raise ValueError(f"Failed to vectorize text data: {e}")
            y_encoded = data_y.map(self.target_label_mapping).values
            try:
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
                grid.fit(x_vectorized, y_encoded)
            except Exception as e:
                logger.error(f"Grid search error: {e}")
                raise ValueError(f"Grid search failed: {e}")
            try:
                y_pred = grid.predict(x_vectorized)
                final_f1 = f1_score(y_encoded, y_pred, average='weighted')
                final_acc = accuracy_score(y_encoded, y_pred)
                logger.info(f"Final Model F1 Score: {final_f1:.4f}")
                logger.info(f"Final Model Accuracy Score: {final_acc:.4f}")
            except Exception as e:
                logger.error(f"Error during final model evaluation: {e}")
                final_f1 = np.nan
                final_acc = np.nan
            self.model = grid
            return {
                'best_params': grid.best_params_,
                'final_f1': final_f1,
                'final_accuracy': final_acc,
                'feature_labels': feature_labels,
                'target_label_mapping': self.target_label_mapping,
                'classifier_type': self.classifier_type.name,
            }

    def predict(
            self,
            data: Union[str, List[str], pd.Series, pd.DataFrame],
            feature_labels: Optional[Union[str, List[str]]] = None,
    ) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if feature_labels is None and self.feature_labels is not None:
            feature_labels = self.feature_labels
        x_vectorized = self._vectorize(data, feature_labels)
        try:
            if self.classifier_type == ClassifierType.BERT:
                preds = self.model.predict(x_vectorized)
                logits = preds.logits
                num_labels = self.model.config.num_labels
                if num_labels > 1:
                    numerical_preds = np.argmax(logits, axis=1)
                else:
                    numerical_preds = (tf.sigmoid(logits) > 0.5).numpy().flatten().astype(int)
            else:
                numerical_preds = self.model.predict(x_vectorized)
            if self.target_label_inverse_mapping:
                return np.array([self.target_label_inverse_mapping[int(idx)] for idx in numerical_preds])
            else:
                return numerical_preds
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Failed to make predictions: {e}")

    def predict_proba(
            self,
            data: Union[str, List[str], pd.Series, pd.DataFrame],
            feature_labels: Optional[Union[str, List[str]]] = None,
    ) -> np.ndarray:
        """Return probability estimates for each class."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if feature_labels is None and self.feature_labels is not None:
            feature_labels = self.feature_labels
        x_vectorized = self._vectorize(data, feature_labels)
        try:
            if self.classifier_type == ClassifierType.BERT:
                preds = self.model.predict(x_vectorized)
                logits = preds.logits
                probs = tf.nn.softmax(logits, axis=1).numpy()
                return probs
            else:
                return self.model.predict_proba(x_vectorized)
        except Exception as e:
            logger.error(f"Error in predict_proba: {e}")
            raise ValueError(f"Failed to get probability estimates: {e}")

    def save(self, filepath: Optional[str] = None):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        if filepath is None:
            os.makedirs('../data/trained', exist_ok=True)
            filepath = f'../data/trained/model-{self.classifier_type.name}.pkl'
        data_to_save = {
            'classifier_type': self.classifier_type,
            'target_label_mapping': self.target_label_mapping,
            'target_label_inverse_mapping': self.target_label_inverse_mapping,
            'feature_labels': self.feature_labels,
            'max_length': self.max_length,
        }
        try:
            if self.classifier_type == ClassifierType.BERT:
                # Save the model and tokenizer using Hugging Face methods.
                model_dir = filepath.replace('.pkl', '')
                os.makedirs(model_dir, exist_ok=True)
                self.model.save_pretrained(model_dir)
                self.bert_tokenizer.save_pretrained(model_dir)
                data_to_save['model_dir'] = model_dir
            else:
                data_to_save['vectorizer'] = self.vectorizer
                data_to_save['model'] = self.model
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise ValueError(f"Failed to save model: {e}")

    @classmethod
    def load(cls, filepath: Optional[str] = None,
             classifier_type: ClassifierType = ClassifierType.SVM) -> 'KnowledgeGraphClassifier':
        """Load a trained model from a file."""
        if filepath is None:
            filepath = f'../data/trained/model-{classifier_type.name}.pkl'
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            instance = cls(classifier_type=data.get('classifier_type', classifier_type))
            instance.target_label_mapping = data.get('target_label_mapping')
            instance.target_label_inverse_mapping = data.get('target_label_inverse_mapping')
            instance.feature_labels = data.get('feature_labels')
            instance.max_length = data.get('max_length')
            if instance.classifier_type == ClassifierType.BERT:
                model_dir = data.get('model_dir')
                if model_dir is None or not os.path.exists(model_dir):
                    raise ValueError("Saved BERT model directory not found.")
                instance.model = TFBertForSequenceClassification.from_pretrained(model_dir)
                instance.bert_tokenizer = BertTokenizer.from_pretrained(model_dir)
            else:
                instance.vectorizer = data.get('vectorizer')
                instance.model = data.get('model')
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError(f"Failed to load model from {filepath}: {e}")


def save_multiple_models(
        models: Dict[str, KnowledgeGraphClassifier],
        training_results: Dict[str, Any],
        filepath: Optional[str] = None
) -> None:
    """Save multiple trained models and their training results to a file."""
    if filepath is None:
        os.makedirs('../data/trained', exist_ok=True)
        filepath = '../data/trained/multiple_models.pkl'
    try:
        with open(filepath, 'wb') as f:
            pickle.dump({'models': models, 'training_results': training_results}, f)
        logger.info(f"Multiple models saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving multiple models: {e}")
        raise ValueError(f"Failed to save multiple models: {e}")


def load_multiple_models(filepath: Optional[str] = None) -> Tuple[Dict[str, KnowledgeGraphClassifier], Dict[str, Any]]:
    """Load multiple trained models and their training results from a file."""
    if filepath is None:
        filepath = '../data/trained/multiple_models.pkl'
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Multiple models loaded from {filepath}")
        return data['models'], data['training_results']
    except Exception as e:
        logger.error(f"Error loading multiple models: {e}")
        raise ValueError(f"Failed to load multiple models from {filepath}: {e}")
