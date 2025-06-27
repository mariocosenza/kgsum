from __future__ import annotations
import os
import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Any, NamedTuple, Self
from collections.abc import Sequence

# --- Ensure reproducibility across all libraries ---
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)
# ---------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import RandomOverSampler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoencoderType(Enum):
    """Supported autoencoder architectures."""
    MLP = auto()
    DEEP = auto()
    BATCHNORM = auto()

class TrainingResult(NamedTuple):
    f1: float
    accuracy: float
    confusion_matrix: np.ndarray
    report: str
    best_params: dict[str, Any]

def parse_feature_value(val: Any) -> str:
    """Flatten lists and parse string-represented lists to space-separated string."""
    if isinstance(val, list):
        return " ".join(map(str, val))
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        import ast
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return " ".join(map(str, parsed))
        except Exception:
            pass
    return str(val)

class AutoencoderDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.data[idx]

class AEMLP(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim), nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

class AEDeep(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim), nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

class AEBatchNorm(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

def get_autoencoder(arch: AutoencoderType, input_dim: int, latent_dim: int = 32) -> nn.Module:
    if arch == AutoencoderType.MLP:
        return AEMLP(input_dim, latent_dim)
    elif arch == AutoencoderType.DEEP:
        return AEDeep(input_dim, latent_dim)
    elif arch == AutoencoderType.BATCHNORM:
        return AEBatchNorm(input_dim, latent_dim)
    else:
        raise ValueError(f"Unknown autoencoder architecture: {arch}")

class CategoryOneHotEncoder(BaseEstimator, TransformerMixin):
    """Robust one-hot encoder, handles unseen labels at test time."""
    def __init__(self) -> None:
        self.encoder: OneHotEncoder | None = None
        self.columns: list[str] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> Self:
        logger.info("[ENC] Fitting OneHotEncoder")
        if isinstance(X, pd.DataFrame):
            self.columns = list(X.columns)
            X_ = X.values
        else:
            self.columns = [f"col_{i}" for i in range(X.shape[1])]
            X_ = X
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder.fit(X_)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        logger.info("[ENC] Transforming data with OneHotEncoder")
        X_ = X.values if isinstance(X, pd.DataFrame) else X
        if self.encoder is None:
            raise RuntimeError("Encoder not fitted!")
        return self.encoder.transform(X_)

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y: Any = None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)

class FeatureTfidfEncoder(BaseEstimator, TransformerMixin):
    """TF-IDF encoding for large token sets, up to 10,000 features."""
    def __init__(self, max_features: int = 10000) -> None:
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            max_features=max_features,
            norm="l2",
            token_pattern=r"(?u)\b\w+\b"
        )
    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> Self:
        logger.info("[ENC] Fitting TfidfVectorizer")
        X_str = self._to_str_series(X)
        self.vectorizer.fit(X_str)
        return self
    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        logger.info("[ENC] Transforming data with TfidfVectorizer")
        X_str = self._to_str_series(X)
        return self.vectorizer.transform(X_str).toarray()
    def fit_transform(self, X: pd.DataFrame | np.ndarray, y: Any = None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)
    def _to_str_series(self, X: pd.DataFrame | np.ndarray) -> list[str]:
        # Always work with a 1D string list/series
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].astype(str).tolist()
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                if X.shape[1] == 1:
                    return X[:, 0].astype(str).tolist()
                else:
                    raise ValueError("TfidfEncoder expects 1D input (one text column)")
            return X.astype(str).tolist()
        else:
            return pd.Series(X).astype(str).tolist()

class AutoencoderEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, latent_dim: int = 32, epochs: int = 20, batch_size: int = 64,
        lr: float = 1e-3, arch: AutoencoderType = AutoencoderType.MLP, verbose: bool = False
    ) -> None:
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.arch = arch
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: nn.Module | None = None
        self.input_dim: int | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> Self:
        logger.info(f"[AE] Fitting Autoencoder ({self.arch.name})")
        X_ = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        self.input_dim = X_.shape[1]
        dataset = AutoencoderDataset(X_)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = get_autoencoder(self.arch, self.input_dim, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if self.verbose and (epoch == 0 or epoch == self.epochs - 1):
                logger.info(f"[{self.arch.name}] Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        logger.info(f"[AE] Transforming data with Autoencoder ({self.arch.name})")
        if self.model is None:
            raise RuntimeError("Autoencoder not fitted!")
        X_ = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        dataset = AutoencoderDataset(X_)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        latents = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                latents.append(self.model.encoder(batch_x).cpu().numpy())
        return np.vstack(latents)

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y: Any = None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)

def build_pipeline(
    arch: AutoencoderType, latent_dim: int = 32, use_tfidf: bool = True, oversample: bool = False
) -> Pipeline:
    logger.info("[PIPELINE] Building model pipeline. TF-IDF: %s | Oversample: %s", use_tfidf, oversample)
    steps: list[tuple[str, Any]] = []
    if oversample:
        steps.append(("oversampler", RandomOverSampler(random_state=42)))
    if use_tfidf:
        steps.append(("tfidf", FeatureTfidfEncoder(max_features=10000)))
    else:
        steps.append(("onehot", CategoryOneHotEncoder()))
    steps.append(("autoencoder", AutoencoderEncoder(latent_dim=latent_dim, arch=arch)))
    steps.append(("classifier", RandomForestClassifier(random_state=42)))
    return Pipeline(steps)

def random_search_params() -> dict[str, Sequence[Any]]:
    return {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [10, 20, 30, None],
        "classifier__min_samples_split": [2, 5, 10],
    }

def train_and_validate_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    arch: AutoencoderType, latent_dim: int = 32,
    use_tfidf: bool = True, oversample: bool = False
) -> tuple[Pipeline, TrainingResult]:
    logger.info("[TRAIN] Starting random search + validation")
    pipe = build_pipeline(arch=arch, latent_dim=latent_dim, use_tfidf=use_tfidf, oversample=oversample)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=random_search_params(),
        n_iter=7, scoring="f1_weighted", n_jobs=-1, cv=3, random_state=42, verbose=1,
        error_score="raise"
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    y_pred = best_pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, zero_division=0)
    logger.info("[TRAIN] Best params: %s", search.best_params_)
    logger.info("[TRAIN] F1: %.4f | Acc: %.4f", f1, acc)
    return best_pipe, TrainingResult(f1, acc, cm, rep, search.best_params_)

def save_models(
    models: dict[str, Pipeline], training_results: dict[str, TrainingResult],
    path: str
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        import pickle
        pickle.dump({
            "models": models,
            "training_results": training_results,
        }, f)
    logger.info("[SAVE] Models saved at: %s", path)

def load_models(path: str) -> tuple[dict[str, Pipeline], dict[str, TrainingResult]]:
    with open(path, "rb") as f:
        import pickle
        data = pickle.load(f)
    logger.info("[LOAD] Models loaded from: %s", path)
    return data["models"], data["training_results"]

def train_autoencoder_models(
    df: pd.DataFrame, feature_columns: list[str], target_label: str,
    arch: AutoencoderType, latent_dim: int, use_tfidf: bool = True, oversample: bool = False
) -> tuple[dict[str, Pipeline], dict[str, TrainingResult]]:
    models: dict[str, Pipeline] = {}
    training_results: dict[str, TrainingResult] = {}
    for feat in feature_columns:
        logger.info("[TRAIN] Training for feature: %s", feat)
        feat_data = df[feat].map(parse_feature_value)
        X = pd.DataFrame({feat: feat_data})
        y = df[target_label].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=0.2, stratify=y, random_state=42
        )
        model, result = train_and_validate_model(
            X_train, y_train, X_test, y_test,
            arch=arch, latent_dim=latent_dim,
            use_tfidf=use_tfidf, oversample=oversample
        )
        models[feat] = model
        training_results[feat] = result
        print(f"\n==== Feature '{feat}' - {arch.name} ====")
        print(result.report)
        print("Confusion Matrix:\n", result.confusion_matrix)
    return models, training_results

def predict_category_majority_vote(
    models: dict[str, Pipeline], processed_data: pd.DataFrame
) -> list[Any]:
    logger.info("[PREDICT] Predicting majority vote for input shape: %s", processed_data.shape)
    all_preds = []
    for feat, model in models.items():
        logger.info("[PREDICT] Predicting with feature: %s", feat)
        feat_data = processed_data[feat].map(parse_feature_value)
        X_feat = pd.DataFrame({feat: feat_data})
        pred = model.predict(X_feat.values)
        all_preds.append(pred)
    all_preds = np.array(all_preds)
    final_preds = []
    for col in all_preds.T:
        vals, counts = np.unique(col, return_counts=True)
        final_preds.append(vals[counts.argmax()])
    return final_preds