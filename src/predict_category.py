import logging

import pandas as pd
import src.pipeline_build
from src.pipeline_build import ClassifierType, majority_vote, predict_category_multi, save_multiple_models, \
    load_multiple_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CategoryPredictor:
    def __init__(self, models, training_results):
        self.models = models
        self.training_results = training_results

    def predict_category(self, processed_data):
        return majority_vote(predict_category_multi(self.models, processed_data))

    @staticmethod
    def get_predictor():
        combined_df = pd.read_json('../data/processed/combined.json')
        feature_columns = ["lab", "lcn", "lpn", "sbj", "dsc"]

        try:
           models, training_results  = load_multiple_models('../data/trained/multiple_models.pkl')
        except Exception as e:
            models, training_results = src.pipeline_build.train_multiple_models(
                combined_df,
                feature_columns,
                target_label="category",
                classifier_type=ClassifierType.SVM
            )
            save_multiple_models(models, training_results)

        logger.info("Global models trained. Training results:")
        for feature, metrics in training_results.items():
            logger.info(f"Feature: {feature}, CV Mean: {metrics['cv_mean']:.3f}, Best Params: {metrics['best_params']}")

        return CategoryPredictor(models, training_results)












