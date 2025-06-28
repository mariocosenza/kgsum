from __future__ import annotations

import logging
import os
import shutil

import pandas as pd

from autoencoder_pipeline import AutoencoderType, load_models, train_autoencoder_models, save_models, \
    predict_category_majority_vote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Robust path resolution ---
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_folder_path():
    return os.path.join(get_project_root(), "data", "trained")

def get_model_file_path():
    return os.path.join(get_data_folder_path(), "multiple_models_autoencoder.pkl")

# --------------------------------

FEATURES: list[str] = ['voc', 'curi', 'puri', 'lcn', 'lpn', 'lab', 'comments', 'tlds']
AE_MODEL: AutoencoderType = AutoencoderType.BATCHNORM
LATENT_DIM: int = 32
TARGET_LABEL: str = "category"
USE_TFIDF: bool = True  # Set to False if you want only OneHot

def main() -> None:
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
            combined_df, FEATURES, TARGET_LABEL, AE_MODEL, LATENT_DIM, use_tfidf=USE_TFIDF, oversample=True
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

    # Example: predict with majority vote on the first 10 rows
    sample = combined_df.iloc[:10]
    preds = predict_category_majority_vote(models, sample)
    print("Predictions (first 10 rows):", preds)

if __name__ == "__main__":
    main()