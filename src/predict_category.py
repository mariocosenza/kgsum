import pandas as pd
import src.pipeline_build
from src.pipeline_build import ClassifierType

combined_df = pd.read_json('../data/processed/combined.json')

feature_columns = ["lab", "lcn", "lpn", "sbj", "dsc"]

models, training_results = src.pipeline_build.train_multiple_models(
    combined_df,
    feature_columns,
    target_label="category",
    classifier_type=ClassifierType.SVM
)

print("Global models trained. Training results:")
for feature, metrics in training_results.items():
    print(f"Feature: {feature}, CV Mean: {metrics['cv_mean']:.3f}, Best Params: {metrics['best_params']}")


def predict_category_multi(processed_data):
    return src.pipeline_build.predict_category_multi(models, processed_data)



