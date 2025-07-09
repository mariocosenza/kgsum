import asyncio
import logging
from os import path

import dataset_extraction.endpoint_lod_service
import dataset_extraction.github_search
import dataset_extraction.zenodo_records_extraction
import dataset_preparation_remote
import lov_data_preparation
from config import Config, Phase
from dataset_preparation import create_local_dataset, create_local_void_dataset
from util import merge_zenodo_sparql, merge_github_sparql

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _labeling():
    if Phase.LABELING in Config.ALLOWED_PHASE:
        if Config.SEARCH_LOD_CLOUD:
            dataset_extraction.endpoint_lod_service.main()
        else:
            if not path.exists("../data/raw/sparql_full_download.csv"):
                raise FileNotFoundError(f'../data/raw/sparql_full_download.csv does not exist')
        if Config.SEARCH_GITHUB:
            dataset_extraction.github_search.main(use_gemini=Config.USE_GEMINI)
        if Config.SEARCH_ZENODO:
           dataset_extraction.zenodo_records_extraction.main(use_gemini=Config.USE_GEMINI)

    if Config.STOP_BEFORE_MERGING:
        logger.info("Stop before merging.")
        exit(1)

    # if path.exists("../data/raw/sparql_full_download.csv"):
    #     if not path.exists("../data/raw/sparql_full_download.csv"):
    #         merge_zenodo_sparql()
    #     if path.exists("../data/raw/sparql_full_download.csv"):
    #         merge_github_sparql()

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


def main():
    Config.init_configuration()
    #_labeling()
    _extraction()

if __name__ == '__main__':
    main()
