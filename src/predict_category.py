import asyncio

import pandas as pd

from src.dataset_preparation import process_file_full_inplace
from src.dataset_preparation_remote import process_endpoint_full_inplace
from src.pipeline_build import train_multiple_models, predict_category_multi
from src.preprocessing import process_all_from_input

combined_df = pd.read_json('../data/processed/combined.json')

feature_columns = ["lab", "lcn", "lpn", "sbj", "dsc"]

models, training_results = train_multiple_models(
    combined_df,
    feature_columns,
    target_label="category"
)

print("Global models trained. Training results:")
for feature, metrics in training_results.items():
    print(f"Feature: {feature}, CV Mean: {metrics['cv_mean']:.3f}, Best Params: {metrics['best_params']}")


async def predict_category_remote_multi(sparql):
    result = await process_endpoint_full_inplace(sparql)

    result = process_all_from_input(result)

    print(predict_category_multi(models, result))
    return predict_category_multi(models, result)


def predict_category_local_multi(file_path):
    result = process_file_full_inplace(file_path)

    result = process_all_from_input(result)

    return predict_category_multi(models, result)


if __name__ == "__main__":
    asyncio.run(predict_category_remote_multi('http://river.styx.org/sparql'))
