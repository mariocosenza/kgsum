import logging
import os
import re
import time
from typing import List, Dict, Any, Set, Optional

import pandas as pd
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

from config import Config
from src.util import merge_dataset, VOC_FILTER, CURI_PURI_FILTER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vocabulary_extraction")

CLEAN_URI = re.compile(r'^([^#]+)?')
IS_URI = re.compile(
    r"^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$")
LOCAL_ENDPOINT_LOV = os.environ['LOCAL_ENDPOINT_LOV']

# Create a global persistent session to reuse connections
session = requests.Session()


def find_tags_from_json(data_frame: pd.DataFrame, voc_filter=Config.USE_FILTER) -> List[str]:
    response_df = pd.DataFrame(columns=['id', 'tags', 'voc', 'category'])
    subject_list = []
    response_cache = {}

    for index, row in data_frame.iterrows():
        all_vocs = []
        all_tags = []

        for voc in set(row['voc']):
            logger.info(f'Vocabulary: {voc}')
            if voc_filter and voc in VOC_FILTER:
                continue
            try:
                if voc not in response_cache:
                    response_lov = session.get(
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
                    response_dict = response_lov.json()
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
            # Create a new row as a dictionary and concatenate it to the DataFrame
            new_row = pd.DataFrame([{
                'id': row['id'],
                'tags': all_tags,
                'voc': all_vocs,
                'category': row['category']
            }])
            response_df = pd.concat([response_df, new_row], ignore_index=True)

    return subject_list


def _get_lov_search_result(uri: str, cache_dict: Dict[str, Any]) -> Optional[frozenset]:
    match = CLEAN_URI.search(uri)
    clean_uri = match.group(1) if match else uri
    if clean_uri not in cache_dict:
        try:
            response_lov = session.get(
                f"https://lov.linkeddata.es/dataset/lov/api/v2/term/search?q={uri}",
                timeout=60
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for term search {uri}: {e}")
            cache_dict[clean_uri] = None
            return None
        if response_lov.status_code == 200:
            try:
                buckets = response_lov.json().get('aggregations', {}) \
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
        cache_dict[clean_uri] = None
        return None
    else:
        return cache_dict[clean_uri]


def find_voc_local(data_frame: pd.DataFrame, voc_filter=Config.USE_FILTER) -> pd.DataFrame:
    sparql = SPARQLWrapper(LOCAL_ENDPOINT_LOV)
    sparql.setReturnFormat(JSON)
    count = 0
    response_df = pd.DataFrame(columns=['id', 'tags', 'voc', 'category'])

    for index, row in data_frame.iterrows():
        all_tags = set()
        all_vocs = []
        for voc in set(row['voc']):
            if voc_filter and voc in VOC_FILTER:
                continue
            if count == 10000:
                time.sleep(60)
                count = 0
            count += 1
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
                    voc_tags = {term['o']['value'] for term in res['results']['bindings']}
                    if voc_tags:
                        all_tags.update(voc_tags)
                        all_vocs.append(voc)
                except Exception as e:
                    logger.error(f'Error processing voc {voc}: {e}')
                    time.sleep(2)
                    continue
        if all_vocs:
            # Create a new row and concatenate it
            new_row = pd.DataFrame([{
                'id': row['id'],
                'tags': list(all_tags),
                'voc': all_vocs,
                'category': row['category']
            }])
            response_df = pd.concat([response_df, new_row], ignore_index=True)
    return response_df


def _process_row(row_column: Set[str], curi_filter=Config.USE_FILTER) -> Optional[List[str]]:
    sparql = SPARQLWrapper(LOCAL_ENDPOINT_LOV)
    sparql.setReturnFormat(JSON)
    all_comments = set()

    for curi in row_column:
        if IS_URI.match(curi):
            if curi_filter and curi in CURI_PURI_FILTER:
                continue
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
                all_comments.update(comments)
                time.sleep(0.1)
            except Exception as e:
                logger.error(f'Error processing uri {curi}: {e}')
                time.sleep(10)
                continue

    return list(all_comments) if all_comments else None


def find_local_curi_puri_comments(data_frame: pd.DataFrame) -> pd.DataFrame:
    response_df = pd.DataFrame(columns=['id', 'curi', 'puri', 'curi_comments', 'puri_comments', 'category'])

    for index, row in data_frame.iterrows():
        logger.info(f'Processing curi and puri in row: {index}')
        curi_set = set(row['curi'])
        puri_set = set(row['puri'])
        curi_comments = _process_row(curi_set)
        puri_comments = _process_row(puri_set)

        # Create new row and concatenate
        new_row = pd.DataFrame([{
            'id': row['id'],
            'curi': list(curi_set),  # Convert set to list for DataFrame storage
            'puri': list(puri_set),  # Convert set to list for DataFrame storage
            'curi_comments': curi_comments,
            'puri_comments': puri_comments,
            'category': row['category']
        }])
        response_df = pd.concat([response_df, new_row], ignore_index=True)

    # Combine comments from curi and puri columns
    response_df['comments'] = response_df.apply(
        lambda x: (x['curi_comments'] or []) + (x['puri_comments'] or []), axis=1
    )
    return response_df


def find_local_curi_puri_comments_combined(uri_list: List[str]) -> List[str]:
    comments = _process_row(set(uri_list))
    return comments if comments else []


def find_voc_local_combined(voc_list: List[str]) -> List[str]:
    sparql = SPARQLWrapper(LOCAL_ENDPOINT_LOV)
    sparql.setReturnFormat(JSON)
    all_tags = set()
    voc_list = set(voc_list)
    count = 0

    for voc in voc_list:
        logger.info(f'Processing voc {voc}')
        if count == 1000:
            time.sleep(60)
            count = 0
        count += 1
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


def _find_voc_tags_from_list(voc_list: List[str]) -> List[str]:
    sparql = SPARQLWrapper(LOCAL_ENDPOINT_LOV)
    sparql.setReturnFormat(JSON)
    count = 0
    tags = set()

    for voc in set(voc_list):
        if count == 10000:
            time.sleep(60)
            count = 0
        count += 1
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
                voc_tags = {term['o']['value'] for term in res['results']['bindings']}
                if voc_tags:
                    tags = tags.union(voc_tags)
            except Exception as e:
                logger.error(f'Error processing voc {voc}: {e}')
                time.sleep(2)
                continue

    return list(tags)


def find_tags_from_list(voc_list: List[str]) -> List[str]:
    try:
        return _find_voc_tags_from_list(voc_list)
    except Exception as e:
        logger.error(f'Error processing voc {voc_list}: {e}')
        return []


def find_comments_from_lists(curi_list: List[str], puri_list: List[str]) -> List[str]:
    all_comments = []
    curi_comments = _process_row(set(curi_list))
    puri_comments = _process_row(set(puri_list))

    if curi_comments:
        all_comments.extend(curi_comments)
    if puri_comments:
        all_comments.extend(puri_comments)

    return all_comments


def find_comments_and_voc_tags(data_frame: pd.DataFrame) -> pd.DataFrame:
    logger.info('Started processing LOV data')
    voc_tags_df = find_voc_local(data_frame)
    comments_df = find_local_curi_puri_comments(data_frame)
    merged_df = pd.merge(comments_df, voc_tags_df, on=['id', 'category'], how='left')
    logger.info('Finished processing LOV data')
    return merged_df


def main():
    merged_df = merge_dataset()
    result_df = find_comments_and_voc_tags(merged_df)
    result_df.to_json('../data/raw/lov_cloud/voc_cmt.json', orient='records')
    session.close()


if __name__ == '__main__':
    main()