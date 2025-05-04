import hashlib
import os
import re

import pandas as pd
from SPARQLWrapper import SPARQLWrapper

# Precompile the regex used to extract the file number
FILE_NUM_REGEX = re.compile(r'(\d+).*\.((?:rdf)|(?:nt)|(?:ttl)|(?:nq))$', re.IGNORECASE)
FILE_STRING_REGEX = re.compile(r'-(.*)\.')

CATEGORIES = {
    'cross_domain', 'geography', 'government', 'life_sciences',
    'linguistics', 'media', 'publications', 'social_networking', 'user_generated'
}
LOD_CATEGORY_NO_MULTIPLE_DOMAIN = {
    'geography', 'government', 'life_sciences',
    'linguistics', 'media', 'publications', 'social_networking'
}

LOD_CATEGORY_NO_USER_DOMAIN = {
    'geography', 'government', 'life_sciences', 'cross_domain',
    'linguistics', 'media', 'publications', 'social_networking'
}


CURI_FILTER = {
    'http://www.w3.org/2002/07/owl',
    'http://www.w3.org/2004/02/skos/core',
    'http://www.w3.org/2000/01/rdf-schema',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns',
    'http://www.w3.org/2000/01/rdf-schema',
    'http://www.w3.org/ns/shacl',
    'http://www.w3.org/ns/prov',
    'http://rdfs.org/ns/void#Dataset'
}

VOC_FILTER = {
    'http://purl.org/dc/terms',
    'http://purl.org/vocab/vann',
    'http://purl.org/dc/elements/1.1',
    'http://schema.org',
    'https://schema.org',
    'http://dbpedia.org/property',
    'http://dbpedia.org/ontology',
    'http://xmlns.com/foaf/0.1',
    'http://example.org',
    'http://yoshimi.sourceforge.net/lv2_plugin',
    'http://rdfs.org/ns/void',
    'http://xmls.com/foaf/0.1'
}

def is_curi_allowed(uri: str) -> bool:
    return not any(url in uri for url in CURI_FILTER)

def is_voc_allowed(uri: str) -> bool:
    return not any(url in uri for url in VOC_FILTER)

def is_endpoint_working(endpoint) -> bool:
    query_string = """
      SELECT ?s ?p ?o
      WHERE {
        ?s ?p ?o
      } LIMIT 1
    """
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


def merge_csv_files(csv1_path, csv2_path, output_csv_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Transform CSV2:
    # Use 'record_link' as 'id' (ignoring 'title' and 'description') and keep 'category'.
    df2_transformed = df2.rename(columns={'record_link': 'id'})[['id', 'category']].copy()
    # Add missing columns with empty strings.
    df2_transformed["download_url"] = ""
    df2_transformed["sparql_url"] = ""
    # Reorder columns to match CSV1.
    df2_transformed = df2_transformed[['id', 'category', 'download_url', 'sparql_url']]

    # Merge the two dataframes vertically.
    merged_df = pd.concat([df1, df2_transformed], ignore_index=True)

    # Write the merged DataFrame to CSV including the index column.
    merged_df.to_csv(output_csv_path, index=True)

    return merged_df


def merge_zenodo_sparql(csv1_path='../data/raw/sparql_full_download.csv',
                        csv2_path='../data/raw/zenodo.csv') -> pd.DataFrame:
    df1 = pd.read_csv(csv1_path, index_col=0)

    # Read CSV2 normally
    df2 = pd.read_csv(csv2_path)

    # Transform CSV2:
    # Rename 'record_link' to 'id' and keep only the 'id' and 'category' columns.
    df2_transformed = df2.rename(columns={'record_link': 'id'})[['id', 'category']].copy()
    # Add missing columns with empty strings.
    df2_transformed["download_url"] = ""
    df2_transformed["sparql_url"] = ""
    # Reorder columns to match CSV1: id, category, download_url, sparql_url.
    df2_transformed = df2_transformed[['id', 'category', 'download_url', 'sparql_url']]

    # Merge the two DataFrames (resetting index so there's only one index column)
    merged_df = pd.concat([df1, df2_transformed], ignore_index=True)

    # Write the merged DataFrame to CSV with one index column
    merged_df.to_csv(csv1_path, index=True)

    return merged_df.drop_duplicates(subset=['id'])


def merge_github_sparql(csv1_path='../data/raw/sparql_full_download.csv',
                        csv2_path='../data/raw/github_unique_repos_with_ttl_nt.csv') -> pd.DataFrame:
    # Read both CSV files
    df1 = pd.read_csv(csv1_path, index_col=0)
    df2 = pd.read_csv(csv2_path)

    # Drop rows in df2 where 'category' is NaN
    df2 = df2.dropna(subset=['category'])

    # Transform df2 to match the format of df1
    df2_transformed = pd.DataFrame({
        'id': df2['repository'],
        'category': df2['category'],
        'download_url': df2['file_url'],
        'sparql_url': ""  # leave sparql_url blank
    })

    # Concatenate the two DataFrames with a fresh index
    merged_df = pd.concat([df1, df2_transformed], ignore_index=True)

    # Save the CSV with the index included (only one index column will be added)
    merged_df.to_csv(csv1_path, index=True)

    return merged_df.drop_duplicates(subset=['id'])


if __name__ == '__main__':
    merge_github_sparql()
