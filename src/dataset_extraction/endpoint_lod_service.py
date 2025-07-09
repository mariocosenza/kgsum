import hashlib
from os import path, makedirs

import pandas as pd
import requests
import logging
from src.util import is_endpoint_working

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("endpoint_lod")

def download_lod_cloud_json_as_csv():
    url = "https://lod-cloud.net/versions/latest/lod-data.json"
    output_path = path.join("..", "data", "raw", "lod-data.json")
    makedirs(path.dirname(output_path), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Downloaded to {output_path}")


def download_dataset():
    df = pd.read_csv('../data/raw/sparql_full_download.csv')
    for index, row in df.iterrows():
        category = row['category']
        if category == "user_generated":
            continue

        download_url = row['download_url']
        # Salta valori nulli o non stringa
        if pd.isna(download_url) or isinstance(download_url, float):
            continue

        # Cast esplicito a str per evitare errori di tipo
        download_url = str(download_url)

        logger.info(f"Downloading {download_url} Number: {index}")
        try:
            response: requests.Response = requests.get(download_url, timeout=600)
        except Exception as e:
            logger.warning(f"Error downloading {download_url}: {e}")
            continue

        if (
            response.status_code == 200 and
            response.headers.get('Content-Type') and
            response.headers['Content-Type'] != 'text/html'
        ):
            dir_path = f'../data/raw/rdf_dump/{category}'
            makedirs(dir_path, exist_ok=True)
            filename = f"{index}-{hashlib.sha256(str(row['id']).encode()).hexdigest()}.rdf"
            file_path = path.join(dir_path, filename)
            try:
                with open(file_path, 'x', encoding='utf-8') as f:
                    f.write(response.text)
            except FileExistsError:
                logger.info(f"File already exists: {file_path}")
            except Exception as e:
                logger.error(f"Error writing file {file_path}: {e}")

def extract_sparql_or_full_download_list():
    frame = pd.read_json('../data/raw/lod-data.json', orient='columns')
    return_list = pd.DataFrame(columns=['id', 'category', 'download_url', 'sparql_url'])
    count = 0
    for i in frame.columns:
        row = frame[i]['full_download']
        count+=1
        logger.info("Extracting " + i + f" Number: {count}" )
        if frame[i]['domain']:
            if len(row) > 0 and row[0]["status"] == 'OK':
                if frame[i]['sparql'] and is_endpoint_working(frame[i]['sparql'][0]['access_url']):
                    return_list.loc[len(return_list)] = [frame[i]['_id'], frame[i]['domain'], row[0]['download_url'], frame[i]['sparql'][0]['access_url']]
                else:
                    return_list.loc[len(return_list)] = [frame[i]['_id'], frame[i]['domain'], row[0]['download_url'], None]
            elif frame[i]['sparql'] and is_endpoint_working(frame[i]['sparql'][0]['access_url']):
                return_list.loc[len(return_list)] = [frame[i]['_id'], frame[i]['domain'], None, frame[i]['sparql'][0]['access_url']]

    return_list.to_csv("../data/raw/sparql_full_download.csv", index=True)

def main():
    if not path.exists("../data/raw/lod-data.json"):
        download_lod_cloud_json_as_csv()
    extract_sparql_or_full_download_list()
    download_dataset()


if __name__ == "__main__":
    main()