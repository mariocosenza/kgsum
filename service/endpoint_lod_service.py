import pandas as pd
import requests
import logging
from src.util import is_endpoint_working

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("endpoint_lod")


def download_dataset(): #expected number of triples 167001612763
    df = pd.read_csv('../data/raw/sparql_full_download.csv')
    for index, row in df.iterrows():
        if not isinstance(row['download_url'], float):
            logger.info("Downloading " + row['download_url'] + f' Number: {index}')
            response = requests.get(row['download_url'], timeout=600)
            if response.status_code == 200 and response.headers['Content-Type'] and response.headers['Content-Type'] != 'text/html':
                open(f'../data/raw/rdf_dump/{row['category']}/{index}.rdf', 'x').write(response.text)

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
