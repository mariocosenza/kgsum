import json
import re
import time

import pandas as pd
import requests

from src.preprocessing import merge_dataset

CLEAN_URI = re.compile(r'^([^#]+)?')


def find_tags_from_json(df: pd.DataFrame):
    response_df = pd.DataFrame(columns=['id', 'tags', 'voc','category'])
    subject_list = []
    response_cache = {}
    for index, row in df.iterrows():
        for voc in set(row['voc']):
            print(f'Vocabulary: {voc}')
            if voc not in response_cache:
                response_lov = requests.get(f"https://lov.linkeddata.es/dataset/lov/api/v2/vocabulary/info?vocab={voc}",
                                            timeout=300)
                response_cache[voc] = response_lov
            else:
                response_lov = response_cache[voc]
            if response_lov.status_code == 200:
                try:
                    response_dict = json.loads(response_lov.text)
                    tags = response_dict['tags']
                    frame_tags = []
                    for tag in tags:
                        if 'Vocabularies' not in tag and 'Metadata' not in tag and 'FRBR' not in tag:
                            subject_list.append(tag)
                            frame_tags.append(tag)
                    response_df.loc[len(response_df)] = {
                        'id': row['id'],
                        'tags': frame_tags,
                        'voc': voc,
                        'category': row['category']
                    }
                except Exception as e:
                    print(e)


    return subject_list

def find_tags_from_json_curi_puri(df: pd.DataFrame):
    response_curi_puri = pd.DataFrame(columns=['id', 'curi', 'puri' 'tags_curi', 'tags_puri', 'tags_voc', 'category'])

    curi_dict = {}
    puri_dict = {}

    for index, row in df.iterrows():
        list_curi_tags = set()
        list_puri_tags = set()

        for curi in row['curi']:
            response_lov = _get_lov_search_result(curi, curi_dict)
            if response_lov != '':
                list_curi_tags.add(response_lov)
            else:
                list_curi_tags.add(None)

        for puri in row['puri']:
            response_lov = _get_lov_search_result(puri, puri_dict)
            if response_lov != '':
                list_puri_tags.add(response_lov)
            else:
                list_puri_tags.add(None)

        response_curi_puri.loc[len(response_curi_puri)] = {
            'id': row['id'],
            'curi': row['curi'],
            'puri': row['puri'],
            'tags_curi': list_curi_tags,
            'tags_puri': list_puri_tags,
            'category': row['category']
        }

    response_curi_puri.to_json('../data/raw/curi_puri_tags.json', orient='records', index=False)

def _get_lov_search_result(uri, cache_dict) -> frozenset | str:
    clean_uri = CLEAN_URI.search(uri)
    clean_uri = clean_uri.group(1)
    if clean_uri not in cache_dict:
        response_lov = requests.get(f"https://lov.linkeddata.es/dataset/lov/api/v2/term/search?q={uri}",
                                 timeout=300)
        if response_lov.status_code == 200:
            key_set = set()
            for keys in json.loads(response_lov.text)['aggregations']['tags']['buckets']:
                key_set.add((keys['key'], keys['doc_count']))
            if len(key_set) > 0:
                print(clean_uri)
                frozenset_set = frozenset(key_set)
                cache_dict[clean_uri] = frozenset_set
                response_lov = frozenset_set
                print(frozenset_set)
                return response_lov
        cache_dict[clean_uri] = ''
        response_lov = ''
    else:
        response_lov = cache_dict[clean_uri]

    return response_lov

df = merge_dataset()
find_tags_from_json(df)