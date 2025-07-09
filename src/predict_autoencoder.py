from __future__ import annotations

import logging
import os
import shutil

import pandas as pd

from autoencoder_pipeline import (
    ClassifierType, load_models, train_autoencoder_models, save_models,
)
from util import get_data_folder_path, get_project_root, get_model_file_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(
    features: list[str] = None,
    classifier: ClassifierType = None,
    latent_dim: int = 32,
    target_label: str = "category",
    use_tfidf: bool = True,
    oversample: bool = True,
) -> None:
    if features is None:
        features = ['voc', 'curi', 'puri', 'lcn', 'lpn', 'lab', 'comments', 'tlds']
    if classifier is None:
        classifier = ClassifierType.BATCHNORM

    directory_path = os.path.join(get_data_folder_path(), 'cache')
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        logger.info("[CLEANUP] Removed old cache directory: %s", directory_path)

    project_root = get_project_root()
    combined_df = pd.read_json(f'{project_root}/data/processed/combined.json')
    logger.info("[DATA] Loaded combined data with shape: %s", combined_df.shape)

    file_path = get_model_file_path()
    if os.path.exists(file_path):
        logger.info(f"[LOAD] Loading pre-trained autoencoder models from: {file_path}")
        models, training_results = load_models(file_path)
    else:
        logger.info("[TRAIN] Training autoencoder models from scratch.")
        models, training_results = train_autoencoder_models(
            combined_df, features, target_label, classifier, latent_dim,
            use_tfidf=use_tfidf, oversample=oversample
        )
        # Ensure directory exists
        os.makedirs(get_data_folder_path(), exist_ok=True)
        save_models(models, training_results, file_path)

    logger.info("[RESULT] Training metrics for each feature:")
    for feature, metrics in training_results.items():
        logger.info(
            "Feature: %s | F1: %.4f | Acc: %.4f | Best params: %s",
            feature, metrics.f1, metrics.accuracy, metrics.best_params
        )

if __name__ == "__main__":
    main()
