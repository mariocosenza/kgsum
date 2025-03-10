import json
import logging
import os
import re

import pandas as pd
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

from src.preprocessing import merge_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vocabulary_extraction")

CLEAN_URI = re.compile(r'^([^#]+)?')
IS_URI = re.compile(
    "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$")
LOCAL_ENDPOINT_LOV = os.environ['LOCAL_ENDPOINT_LOV']


def find_tags_from_json(data_frame: pd.DataFrame):
    response_df = pd.DataFrame(columns=['id', 'tags', 'voc', 'category'])
    subject_list = []
    response_cache = {}
    for index, row in data_frame.iterrows():
        all_vocs = []
        all_tags = []

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

                    if frame_tags:
                        all_vocs.append(voc)
                        all_tags.extend(frame_tags)
                except Exception as e:
                    print(e)

        if all_vocs:
            response_df.loc[len(response_df)] = {
                'id': row['id'],
                'tags': all_tags,
                'voc': all_vocs,
                'category': row['category']
            }

    return subject_list


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


def find_voc_local(data_frame: pd.DataFrame):
    sparql = SPARQLWrapper(LOCAL_ENDPOINT_LOV)
    sparql.setReturnFormat(JSON)
    response_df = pd.DataFrame(columns=['id', 'tags', 'voc', 'category'])

    for index, row in data_frame.iterrows():
        all_tags = set()
        all_vocs = []

        for voc in row['voc']:
            logger.info(f'Processing voc {voc}')
            try:
                sparql.setQuery(f"""
                PREFIX dcat: <http://www.w3.org/ns/dcat#>
                SELECT ?o WHERE {{
                    <{voc}> dcat:keyword ?o .
                }} LIMIT 10
                """)
                res = sparql.query().convert()

                voc_tags = {term['o']['value'] for term in res['results']['bindings']}
                if voc_tags:
                    all_tags.update(voc_tags)
                    all_vocs.append(voc)

            except Exception as e:
                logger.info(f'Invalid uri: {e}')
                pass

        if all_vocs:
            response_df.loc[len(response_df)] = {
                'id': row['id'],
                'tags': all_tags,
                'voc': all_vocs,
                'category': row['category']
            }

    return response_df


def _process_row(row_column):
    sparql = SPARQLWrapper(LOCAL_ENDPOINT_LOV)
    sparql.setReturnFormat(JSON)
    all_comments = list()

    for curi in row_column:
        if IS_URI.match(curi):
            try:
                sparql.setQuery(f"""
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        SELECT ?o WHERE {{
                        <{curi}> rdfs:comment ?o
                        FILTER(langMatches(lang(?o), "en"))
                        }} LIMIT 5""")

                res = sparql.query().convert()
                comments = {term['o']['value'] for term in res['results']['bindings']}
                all_comments.append(comments)
            except Exception as e:
                logger.info(f'Invalid uri: {e}')
                pass

    return all_comments if all_comments else None


def find_local_curi_puri_comments(data_frame: pd.DataFrame):
    response_df = pd.DataFrame(columns=['id', 'curi', 'puri', 'curi_comments', 'puri_comments', 'category'])
    for index, row in data_frame.iterrows():
        logger.info(f'Processing curi and puri in row: {index}')
        response_df.loc[len(response_df)] = {
            'id': row['id'],
            'curi': row['curi'],
            'puri': row['puri'],
            'curi_comments': _process_row(row['curi']),
            'puri_comments': _process_row(row['puri']),
            'category': row['category']
        }

    response_df['comments'] = response_df['curi_comments'] +  response_df['puri_comments']
    return response_df


def find_local_curi_puri_comments_combined(uri_list: list) -> list:
    comments = _process_row(uri_list)
    return list(comments) if comments else []


def find_voc_local_combined(voc_list: list) -> list:
    sparql = SPARQLWrapper(LOCAL_ENDPOINT_LOV)
    sparql.setReturnFormat(JSON)
    all_tags = set()

    for voc in voc_list:
        logger.info(f'Processing voc {voc}')
        try:
            sparql.setQuery(f"""
            PREFIX dcat: <http://www.w3.org/ns/dcat#>
            SELECT ?o WHERE {{
                <{voc}> dcat:keyword ?o .
            }} LIMIT 10
            """)
            res = sparql.query().convert()
            voc_tags = {term['o']['value'] for term in res['results']['bindings']}
            if voc_tags:
                all_tags.update(voc_tags)
        except Exception as e:
            logger.info(f'Invalid uri: {e}')
            continue

    return list(all_tags)


def find_comments_and_voc_tags(data_frame: pd.DataFrame) -> pd.DataFrame:
    logger.info('Started processing lov data')
    voc_tags_df = find_voc_local(data_frame)
    comments_df = find_local_curi_puri_comments(data_frame)

    merged_df = pd.merge(comments_df, voc_tags_df, on=['id', 'category'], how='left')
    logger.info('Finished processing lov data')
    return merged_df


def main():
    find_comments_and_voc_tags(merge_dataset()).to_json('../data/raw/lov_cloud/voc_cmt.json')


if __name__ == '__main__':
    main()
