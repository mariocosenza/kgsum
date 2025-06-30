import logging
import os
import shutil
import pandas as pd

import src.pipeline_build
from src.pipeline_build import ClassifierType, majority_vote, predict_category_multi, save_multiple_models, \
    load_multiple_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Robust path resolution ---
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_folder_path():
    return os.path.join(get_project_root(), 'data', 'trained')

def get_model_file_path():
    return os.path.join(get_data_folder_path(), 'multiple_models.pkl')

# --------------------------------

class CategoryPredictor:
    def __init__(self, models, training_results):
        self.models = models
        self.training_results = training_results

    def predict_category(self, processed_data):
        return majority_vote(predict_category_multi(self.models, processed_data))

    @staticmethod
    def get_predictor(classifier=ClassifierType.NAIVE_BAYES, feature_columns: list[str] = None, oversample = True):
        combined_df_path = os.path.join(get_project_root(), 'data', 'processed', 'combined.json')
        if os.path.exists(combined_df_path):
            combined_df = pd.read_json(combined_df_path)
        else:
            combined_df = pd.DataFrame()
        if feature_columns is None:
            feature_columns = ["curi"]
        file_path = get_model_file_path()
        logger.info(f"Looking for trained model at: {file_path}")
        try:
            models, training_results = load_multiple_models(file_path)
        except Exception as e:
            logger.warning(f"Loading models failed: {e}. Retraining models...")
            models, training_results = src.pipeline_build.train_multiple_models(
                combined_df,
                feature_columns,
                target_label="category",
                classifier_type=classifier,
                oversample=oversample
            )
            # Ensure directory exists
            os.makedirs(get_data_folder_path(), exist_ok=True)
            save_multiple_models(models, training_results, file_path)

        logger.info("Global models trained/loaded. Training results:")
        for feature, metrics in training_results.items():
            logger.info(f'Feature: {feature}, Best Params: {metrics}')

        return CategoryPredictor(models, training_results)


if __name__ == "__main__":
    directory_path = os.path.join(get_data_folder_path(), 'cache')
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    PREDICTOR = CategoryPredictor.get_predictor(
        classifier=ClassifierType.NAIVE_BAYES,
        feature_columns=['voc', 'curi', 'puri', 'lcn', 'lpn', 'tlds'],
        oversample=True
    )