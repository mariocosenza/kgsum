import asyncio
import logging
from os import path, makedirs
import shutil
import dataset_extraction.endpoint_lod_service
import dataset_extraction.github_search
import dataset_extraction.zenodo_records_extraction
import dataset_preparation_remote
import predict_autoencoder
import preprocessing
import lov_data_preparation
from config import Config, Phase, ClassifierType
from dataset_preparation import create_local_dataset, create_local_void_dataset
from generate_profile import generate_profile_from_store
from predict_category import CategoryPredictor
from util import merge_zenodo_sparql, merge_github_sparql, get_data_folder_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _labeling():
    if not path.exists('../data'):
        makedirs('../data')
    if not path.exists('../data/processed'):
        makedirs('../data/processed')
    if not path.exists('../data/trained'):
        makedirs('../data/trained')
    if not path.exists('../data/raw'):
        makedirs('../data/raw')

    if Phase.LABELING in Config.ALLOWED_PHASE:
        if Config.SEARCH_LOD_CLOUD:
            dataset_extraction.endpoint_lod_service.main()
        else:
            if not path.exists("../data/raw/sparql_full_download.csv"):
                raise FileNotFoundError(f'../data/raw/sparql_full_download.csv does not exist')
        if Config.SEARCH_GITHUB:
            dataset_extraction.github_search.main(use_gemini=Config.USE_GEMINI)
            if Config.STOP_BEFORE_MERGING:
                logger.info("Stop before merging.")
                exit(1)
            merge_github_sparql()
        if Config.SEARCH_ZENODO:
           dataset_extraction.zenodo_records_extraction.main(use_gemini=Config.USE_GEMINI)
           if Config.STOP_BEFORE_MERGING:
               logger.info("Stop before merging.")
               exit(1)
           merge_zenodo_sparql()

def _extraction():
    if Phase.EXTRACTION in Config.ALLOWED_PHASE:
        if Config.EXTRACT_SPARQL:
            asyncio.run(dataset_preparation_remote.main_normal())
            asyncio.run(dataset_preparation_remote.main_void())
        for i in range(Config.LIMIT):
            offset = (i + 1) * Config.START_OFFSET
            create_local_dataset(offset=offset, limit=offset + Config.SAVE_RANGE)
            create_local_void_dataset(offset=offset, limit=offset + Config.SAVE_RANGE)
        if Config.QUERY_LOV:
            lov_data_preparation.main()

def _processing():
    if Phase.PROCESSING in Config.ALLOWED_PHASE:
        preprocessing.main(use_ner=Config.USE_NER, enable_filter=Config.USE_FILTER)

def _training():
    if Phase.TRAINING in Config.ALLOWED_PHASE:
        if Config.CLASSIFIER in [ClassifierType.SVM, ClassifierType.NAIVE_BAYES, ClassifierType.KNN, ClassifierType.J48, ClassifierType.MISTRAL]:
            directory_path = path.join(get_data_folder_path(), 'cache')
            if path.exists(directory_path):
                shutil.rmtree(directory_path)
            CategoryPredictor.get_predictor(
                classifier=Config.CLASSIFIER,
                feature_columns=Config.FEATURES,
                oversample=Config.OVERSAMPLE
            )
        else:
            predict_autoencoder.main(classifier=Config.CLASSIFIER, use_tfidf=Config.USE_TF_IDF_AUTOENCODER, oversample=Config.OVERSAMPLE)

def _profile():
    if Phase.STORE in Config.ALLOWED_PHASE:
        asyncio.run(generate_profile_from_store())

def main():
    Config.init_configuration()
    _labeling()
    _extraction()
    _processing()
    _training()
    _profile()

if __name__ == '__main__':
    main()
