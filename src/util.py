import hashlib
import os
import re

import pandas as pd
from SPARQLWrapper import SPARQLWrapper

# Precompile the regex used to extract the file number
FILE_NUM_REGEX = re.compile(r'(\d+).*\.rdf$')
FILE_STRING_REGEX = re.compile(r'-(.*)\.')

CATEGORIES = {
    'cross_domain', 'geography', 'government', 'life_sciences',
    'linguistics', 'media', 'publications', 'social_networking', 'user_generated'
}
LOD_CATEGORY = {
    'geography', 'government', 'life_sciences',
    'linguistics', 'media', 'publications', 'social_networking'
}




def is_endpoint_working(endpoint) -> bool:
    query_string = """
      SELECT ?s ?p ?o
   WHERE {
      ?s ?p ?o
   } LIMIT 1"""
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query_string)
    sparql.setTimeout(120)
    try:
        result = sparql.query()
        str_header = result.response.headers.as_string()
        if result.response.status >= 310:
            return False

        if 'text/plain' in str_header or 'text/html' in str_header or 'application/octet-stream' in str_header:
            return False
        result.convert()
        return True
    except Exception as e:
        return False


def match_file_lod(file, limit, offset, lod_frame) -> int | None:
    match = FILE_NUM_REGEX.match(file)
    if not match:
        return None
    file_num = int(match.group(1))
    if file_num < offset or file_num > limit:
        return None

    match = FILE_STRING_REGEX.search(file)
    num = -1
    if match:
        extracted_string = match.group(1)
        for index, id_frame in enumerate(lod_frame['id']):
            if extracted_string == hashlib.sha256(id_frame.encode()).hexdigest():
                num = index
                return num
    if num == -1:
        return None


def rename_new_file(offset):
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    for category in CATEGORIES:
        directory = f'../data/raw/rdf_dump/{category}'
        for file in os.listdir(directory):
            path = f'../data/raw/rdf_dump/{category}/{file}'
            match = FILE_NUM_REGEX.search(path)
            if match:
                file_num = int(match.group(1))
                if file_num > offset:
                    os.rename(path,
                              f'../data/raw/rdf_dump/{category}/{file_num}-{hashlib.sha256(lod_frame['id'][file_num].encode()).hexdigest()}.rdf')
