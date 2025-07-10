import hashlib
import logging
import os
import re

import pandas as pd
from SPARQLWrapper import SPARQLWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Robust path resolution ---
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_folder_path():
    return os.path.join(get_project_root(), 'data', 'trained')


def get_model_file_path():
    return os.path.join(get_data_folder_path(), 'multiple_models.pkl')


FILE_NUM_REGEX = re.compile(r'^(\d+)[^.]*\.(?:rdf|nt|ttl|nq)$', re.IGNORECASE)
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

FILTER_DATA = pd.read_json(
    os.path.join(get_project_root(), 'src', 'filter', 'filter.json'),
    typ='series'
)

CURI_PURI_FILTER = set(FILTER_DATA['CURI_PURI_FILTER'])
VOC_FILTER = set(FILTER_DATA['VOC_FILTER'])


def is_curi_allowed(uri: str) -> bool:
    for url in CURI_PURI_FILTER:
        if url in uri:
            logging.debug(uri)
            return False
    return True


def is_voc_allowed(uri: str) -> bool:
    for url in VOC_FILTER:
        if url in uri:
            return False
    return True


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
    except Exception as _:
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


def _merge(df1, df2, output_csv_path) -> pd.DataFrame:
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

    merged_df = merged_df.drop_duplicates(subset=['id'])
    # Write the merged DataFrame to CSV including the index column.
    merged_df.to_csv(output_csv_path, index=True)

    return merged_df


def merge_csv_files(csv1_path, csv2_path, output_csv_path):
    df1 = pd.read_csv(csv1_path)

    return _merge(df1, pd.read_csv(csv2_path), output_csv_path)


def merge_zenodo_sparql(csv1_path='../data/raw/sparql_full_download.csv',
                        csv2_path='../data/raw/zenodo.csv') -> pd.DataFrame:
    df1 = pd.read_csv(csv1_path, index_col=0)

    return _merge(df1, pd.read_csv(csv2_path), csv1_path)


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
    merged_df = merged_df.drop_duplicates(subset=['id'])
    merged_df.to_csv(csv1_path, index=True)

    return merged_df


def merge_dump_sparql(csv1_path='../data/raw/graphs.csv',
                      csv2_path='../data/raw/datasetsAndCategories.csv') -> pd.DataFrame:
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    output_df = pd.DataFrame(columns=['id', 'category', 'download_url', 'sparql_url', 'graphs_uri'])

    for index, row in df2.iterrows():
        uri = []
        for i, line in df1.iterrows():
            if row["id"][6:] in line["g"]:  # should use binary search
                uri.append(line["g"])
        row['graphs_uri'] = uri
        output_df.loc[len(output_df)] = row
        print(output_df)

    output_df.to_csv('../data/raw/graphs_with_uri.csv', index=True)
    return df2


DATA_DIR = "../data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


def merge_dataset() -> pd.DataFrame:
    local_frames: list[pd.DataFrame] = []
    remote_frames: list[pd.DataFrame] = []
    local_path = os.path.join(RAW_DIR, "local")
    for fname in os.listdir(local_path):
        if "local_feature_set" in fname and fname.endswith(".json"):
            fullpath = os.path.join(local_path, fname)
            try:
                local_frames.append(pd.read_json(fullpath))
            except Exception as exc:
                logger.error("Failed to read %s: %s", fullpath, exc)
    remote_path = os.path.join(RAW_DIR, "remote")
    for fname in os.listdir(remote_path):
        if "remote_feature_set" in fname and fname.endswith(".json"):
            fullpath = os.path.join(remote_path, fname)
            try:
                remote_frames.append(pd.read_json(fullpath))
            except Exception as exc:
                logger.error("Failed to read %s: %s", fullpath, exc)
    df_local = pd.concat(local_frames, ignore_index=True) if local_frames else pd.DataFrame()
    df_remote = pd.concat(remote_frames, ignore_index=True) if remote_frames else pd.DataFrame()
    merged = pd.concat([df_local, df_remote], ignore_index=True)
    if "id" in merged.columns:
        merged = merged.drop_duplicates(subset="id", keep="last")
    return merged


def merge_void_dataset() -> pd.DataFrame:
    local_frames: list[pd.DataFrame] = []
    remote_frames: list[pd.DataFrame] = []
    local_path = os.path.join(RAW_DIR, "local")
    for fname in os.listdir(local_path):
        if "local_void_feature_set" in fname and fname.endswith(".json"):
            fullpath = os.path.join(local_path, fname)
            try:
                local_frames.append(pd.read_json(fullpath))
            except Exception as exc:
                logger.error("Failed to read void file %s: %s", fullpath, exc)
    remote_path = os.path.join(RAW_DIR, "remote")
    for fname in os.listdir(remote_path):
        if "remote_void_feature_set" in fname and fname.endswith(".json"):
            fullpath = os.path.join(remote_path, fname)
            try:
                remote_frames.append(pd.read_json(fullpath))
            except Exception as exc:
                logger.error("Failed to read void file %s: %s", fullpath, exc)
    df_local = pd.concat(local_frames, ignore_index=True) if local_frames else pd.DataFrame()
    df_remote = pd.concat(remote_frames, ignore_index=True) if remote_frames else pd.DataFrame()
    merged = pd.concat([df_local, df_remote], ignore_index=True)
    if "id" in merged.columns:
        merged = merged.drop_duplicates(subset="id", keep="last")
    return merged


if __name__ == '__main__':
    merge_zenodo_sparql()
    merge_github_sparql()
