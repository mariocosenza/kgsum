import logging
import os.path

from google import genai

import dataset_extraction.github_search

from dataset_extraction.github_search import download_and_predict
from dataset_extraction.zenodo_records_extraction import process_zenodo_records_with_download, get_zenodo_records
from service.endpoint_lod_service import extract_sparql_or_full_download_list, download_dataset
from util import merge_github_sparql, merge_zenodo_sparql

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info(
        'If you like to use Gemini make sure to set the API KEY as env variables GEMINI_API_KEY and GEMINI_API_KEY_2')

    base_folder = '..'
    use_ollama = input('Write S to use OLLAMA')
    client = None
    if not use_ollama:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    if use_ollama.lower() in 'S':
        use_ollama = True
    if not os.path.isfile(f'{base_folder}/data/raw/sparql_full_download.csv'):
        logger.info('Extracting and downloading the available dump from LODCloud')
        extract_sparql_or_full_download_list()
        download_dataset()
    if not os.path.isfile(f'{base_folder}/data/raw/github_unique_repos_with_ttl_nt.csv'):
        user_in = input('Write S to download GitHub rdf dump. Write L to get only the result list')
        if user_in.lower() == 's':
            logger.info('Downloading dump from GitHub')
            download_folder = f"{base_folder}/data/raw/rdf_dump"
            download_and_predict(client, download_folder, use_ollama)
        elif user_in.lower() == 'l':
            logger.info('Creating a list of available repository with ttl, nt or nq data')
            dataset_extraction.github_search.main()
            return
        else:
            return
        merge_github_sparql()
    if not os.path.isfile(f'{base_folder}/data/raw/zenodo_with_files.csv'):
        download_folder = f"{base_folder}/data/raw/rdf_dump"
        output_csv_path = f"{base_folder}/data/raw/zenodo_with_files.csv"
        user_in = input('Write S to download Zenodo rdf dump. Write L to get only the result list')
        if user_in.lower() == 's':
            logger.info('Downloading Zenodo rdf dump')
            process_zenodo_records_with_download(client, download_folder, output_csv_path, use_ollama=use_ollama)
        elif user_in.lower() == 'l':
            logger.info('Creating a list of available repository in Zenodo with ttl, nt or nq data')
            get_zenodo_records(client, use_ollama=use_ollama).to_csv(output_csv_path, index=False)
            return
        else:
            return
        merge_zenodo_sparql()




if __name__ == '__main__':
    main()
