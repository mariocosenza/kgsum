import json
import logging
import os
import re
import time

import pandas as pd
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

from src.preprocessing import merge_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vocabulary_extraction")

CLEAN_URI = re.compile(r'^([^#]+)?')
IS_URI = re.compile(
    r"^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$")
LOCAL_ENDPOINT_LOV = os.environ['LOCAL_ENDPOINT_LOV']


def find_tags_from_json(data_frame: pd.DataFrame):
    response_df = pd.DataFrame(columns=['id', 'tags', 'voc', 'category'])
    subject_list = []
    response_cache = {}
    for index, row in data_frame.iterrows():
        all_vocs = []
        all_tags = []
        for voc in set(row['voc']):
            logger.info(f'Vocabulary: {voc}')
            try:
                if voc not in response_cache:
                    response_lov = requests.get(
                        f"https://lov.linkeddata.es/dataset/lov/api/v2/vocabulary/info?vocab={voc}",
                        timeout=60
                    )
                    response_cache[voc] = response_lov
                else:
                    response_lov = response_cache[voc]
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for vocabulary {voc}: {e}")
                continue

            if response_lov.status_code == 200:
                try:
                    response_dict = json.loads(response_lov.text)
                    tags = response_dict.get('tags', [])
                    frame_tags = []
                    for tag in tags:
                        if 'Vocabularies' not in tag and 'Metadata' not in tag and 'FRBR' not in tag:
                            subject_list.append(tag)
                            frame_tags.append(tag)
                    if frame_tags:
                        all_vocs.append(voc)
                        all_tags.extend(frame_tags)
                except Exception as e:
                    logger.error(f"Error parsing JSON for vocabulary {voc}: {e}")

        if all_vocs:
            response_df.loc[len(response_df)] = {
                'id': row['id'],
                'tags': all_tags,
                'voc': all_vocs,
                'category': row['category']
            }
    return subject_list


def _get_lov_search_result(uri, cache_dict) -> frozenset | str:
    match = CLEAN_URI.search(uri)
    clean_uri = match.group(1) if match else uri
    if clean_uri not in cache_dict:
        try:
            response_lov = requests.get(
                f"https://lov.linkeddata.es/dataset/lov/api/v2/term/search?q={uri}",
                timeout=60
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for term search {uri}: {e}")
            cache_dict[clean_uri] = ''
            return ''
        if response_lov.status_code == 200:
            try:
                buckets = json.loads(response_lov.text).get('aggregations', {})\
                    .get('tags', {}).get('buckets', [])
                key_set = {(item['key'], item['doc_count']) for item in buckets}
                if key_set:
                    logger.info(clean_uri)
                    frozenset_set = frozenset(key_set)
                    cache_dict[clean_uri] = frozenset_set
                    logger.info(frozenset_set)
                    return frozenset_set
            except Exception as e:
                logger.error(f"Error parsing term search for {uri}: {e}")
        cache_dict[clean_uri] = ''
        return ''
    else:
        return cache_dict[clean_uri]


def find_voc_local(data_frame: pd.DataFrame):
    sparql = SPARQLWrapper(LOCAL_ENDPOINT_LOV)
    sparql.setReturnFormat(JSON)
    response_df = pd.DataFrame(columns=['id', 'tags', 'voc', 'category'])
    for index, row in data_frame.iterrows():
        all_tags = set()
        all_vocs = []
        for voc in set(row['voc']):
            logger.info(f'Processing voc {voc}')
            if IS_URI.match(voc):
                try:
                    sparql.setQuery(f"""
                    PREFIX dcat: <http://www.w3.org/ns/dcat#>
                    SELECT ?o WHERE {{
                         <{voc}> dcat:keyword ?o .
                    }} LIMIT 10
                    """)
                    res = sparql.query().convert()
                    # Wait a short moment between queries
                    voc_tags = {term['o']['value'] for term in res['results']['bindings']}
                    if voc_tags:
                        all_tags.update(voc_tags)
                        all_vocs.append(voc)
                except Exception as e:
                    logger.error(f'Error processing voc {voc}: {e}')
                    time.sleep(2)
                    continue
        if all_vocs:
            response_df.loc[len(response_df)] = {
                'id': row['id'],
                'tags': list(all_tags),
                'voc': all_vocs,
                'category': row['category']
            }
    return response_df


def _process_row(row_column: set) -> list[str] | None:
    sparql = SPARQLWrapper(LOCAL_ENDPOINT_LOV)
    sparql.setReturnFormat(JSON)
    all_comments = []
    for curi in row_column:
        if IS_URI.match(curi):
            try:
                sparql.setQuery(f"""
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        SELECT ?o WHERE {{
                        <{curi}> rdfs:comment ?o
                        FILTER(langMatches(lang(?o), "en"))
                        }} LIMIT 5
                        """)
                res = sparql.query().convert()
                comments = {term['o']['value'] for term in res['results']['bindings']}
                all_comments.append(list(comments))
            except Exception as e:
                logger.error(f'Error processing uri {curi}: {e}')
                time.sleep(2)
                continue
    return all_comments if all_comments else None


def find_local_curi_puri_comments(data_frame: pd.DataFrame):
    response_df = pd.DataFrame(columns=['id', 'curi', 'puri', 'curi_comments', 'puri_comments', 'category'])
    for index, row in data_frame.iterrows():
        logger.info(f'Processing curi and puri in row: {index}')
        curi_set = set(row['curi'])
        puri_set = set(row['puri'])
        curi_comments = _process_row(curi_set)
        puri_comments = _process_row(puri_set)
        response_df.loc[len(response_df)] = {
            'id': row['id'],
            'curi': curi_set,
            'puri': puri_set,
            'curi_comments': curi_comments,
            'puri_comments': puri_comments,
            'category': row['category']
        }
    # Combine comments from curi and puri columns
    response_df['comments'] = response_df.apply(
        lambda x: (x['curi_comments'] or []) + (x['puri_comments'] or []), axis=1
    )
    return response_df


def find_local_curi_puri_comments_combined(uri_list: list) -> list:
    comments = _process_row(uri_list)
    return list(comments) if comments else []


def find_voc_local_combined(voc_list: list) -> list:
    sparql = SPARQLWrapper(LOCAL_ENDPOINT_LOV)
    sparql.setReturnFormat(JSON)
    all_tags = set()
    voc_list = set(voc_list)
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
            time.sleep(1)
            voc_tags = {term['o']['value'] for term in res['results']['bindings']}
            if voc_tags:
                all_tags.update(voc_tags)
        except Exception as e:
            logger.error(f'Error processing voc {voc}: {e}')
            time.sleep(2)
            continue
    return list(all_tags)


def find_comments_and_voc_tags(data_frame: pd.DataFrame) -> pd.DataFrame:
    logger.info('Started processing LOV data')
    voc_tags_df = find_voc_local(data_frame)
    comments_df = find_local_curi_puri_comments(data_frame)
    merged_df = pd.merge(comments_df, voc_tags_df, on=['id', 'category'], how='left')
    logger.info('Finished processing LOV data')
    return merged_df


def main():
    merged_df = merge_dataset()  # merge_dataset() should return a DataFrame
    result_df = find_comments_and_voc_tags(merged_df)
    result_df.to_json('../data/raw/lov_cloud/voc_cmt.json', orient='records')


if __name__ == '__main__':
    main()
