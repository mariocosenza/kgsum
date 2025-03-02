import pandas as pd
import os

from src.dataset_preparation_remote import process_endpoint_full_inplace
from src.pipeline_build import KnowledgeGraphClassifier
from src.preprocessing import process_label_feature

current_dir = os.path.dirname(os.path.abspath(__file__))

model = KnowledgeGraphClassifier()
model_data_path = os.path.join(current_dir, '..', 'data', 'processed', 'lab_lcn_lpn.json')
model.train(pd.read_json(model_data_path), 'lab')

async def predict_category_remote(sparql):
    result = await process_endpoint_full_inplace(sparql)
    processed = process_label_feature(list(result['label']))
    return model.predict(processed)[0]

def predict_category_local(file_path):
    result = process_endpoint_full_inplace(file_path)
