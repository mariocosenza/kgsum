import logging
import re
from os import listdir
from typing import Dict, List, Tuple, Any, Union

import pandas as pd
import spacy
from spacy import Language
from spacy_langdetect import LanguageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a spaCy language detector factory
@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()

# Load the main spaCy pipeline (for language detection)
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector", last=True)

# Load language-specific pipelines
pipeline_dict = {}
try:
    pipeline_dict["en"] = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error("Error loading English pipeline: %s", e)
try:
    pipeline_dict["it"] = spacy.load("it_core_news_sm")
except Exception as e:
    logger.error("Error loading Italian pipeline: %s", e)
try:
    pipeline_dict["es"] = spacy.load("es_core_news_sm")
except Exception as e:
    logger.error("Error loading Spanish pipeline: %s", e)
try:
    pipeline_dict["de"] = spacy.load("de_core_news_sm")
except Exception as e:
    logger.error("Error loading German pipeline: %s", e)
try:
    pipeline_dict["fr"] = spacy.load("fr_core_news_sm")
except Exception as e:
    logger.error("Error loading French pipeline: %s", e)

try:
    fallback_pipeline = spacy.load("xx_ent_wiki_sm")
except Exception as e:
    logger.error("Error loading multilingual pipeline: %s", e)
    fallback_pipeline = pipeline_dict.get("en")

DATA_DIR = "../data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"

URI_FRAGMENT_PATTERN = re.compile(r'[#/]([^#/]+)$')
URI_NAMESPACE_PATTERN = re.compile(r'^(.*?)[#/][^#/]+$')
TLD_PATTERN = re.compile(r'^(?:https?://)?(?:www\.)?([^/]+)')


def analyze_uri(uri: str) -> Dict[str, str]:
    result = {"namespace": "", "local_name": "", "tld": ""}
    if not uri or not isinstance(uri, str):
        return result
    fragment_match = URI_FRAGMENT_PATTERN.search(uri)
    if fragment_match:
        result["local_name"] = fragment_match.group(1)
    namespace_match = URI_NAMESPACE_PATTERN.search(uri)
    if namespace_match:
        result["namespace"] = namespace_match.group(1)
    else:
        result["namespace"] = uri
    tld_match = TLD_PATTERN.search(uri)
    if tld_match:
        result["tld"] = tld_match.group(1)
    return result


def process_text(text: str) -> str:
    if not text or not isinstance(text, str) or len(text) > 100000:
        return ""
    try:
        doc = nlp(text)
        lang = doc._.language.get("language", "en")

        chosen_nlp = pipeline_dict.get(lang, fallback_pipeline)
        docs = list(chosen_nlp.pipe([text]))
        return docs[0].text if docs else ""
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return text

def find_language(text: str) -> str:
    try:
        doc = nlp(text)
        return doc._.language.get("language", "en")
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return text


def normalize_text_list(text_list) -> str:
    if not text_list:
        return ""
    if isinstance(text_list, str):
        return text_list
    return " ".join(str(word) for word in text_list if word is not None)


def merge_dataset() -> pd.DataFrame:
    local_frames = []
    for file in listdir(f"{RAW_DIR}/local"):
        if "local_feature_set" in file:
            try:
                local_frames.append(pd.read_json(f"{RAW_DIR}/local/{file}"))
            except Exception as e:
                logger.error(f"Error reading file {file}: {e}")
    remote_frames = []
    for file in listdir(f"{RAW_DIR}/remote"):
        if "remote_feature_set" in file:
            try:
                remote_frames.append(pd.read_json(f"{RAW_DIR}/remote/{file}"))
            except Exception as e:
                logger.error(f"Error reading file {file}: {e}")
    df_local = pd.concat(local_frames, ignore_index=True) if local_frames else pd.DataFrame()
    df_remote = pd.concat(remote_frames, ignore_index=True) if remote_frames else pd.DataFrame()
    merged_df = pd.concat([df_local, df_remote], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset="id", keep="last")
    return merged_df


def merge_void_dataset() -> pd.DataFrame:
    local_frames = []
    for file in listdir(f"{RAW_DIR}/local"):
        if "local_void_feature_set" in file:
            try:
                local_frames.append(pd.read_json(f"{RAW_DIR}/local/{file}"))
            except Exception as e:
                logger.error(f"Error reading void file {file}: {e}")
    remote_frames = []
    for file in listdir(f"{RAW_DIR}/remote"):
        if "remote_void_feature_set" in file:
            try:
                remote_frames.append(pd.read_json(f"{RAW_DIR}/remote/{file}"))
            except Exception as e:
                logger.error(f"Error reading void file {file}: {e}")
    df_local = pd.concat(local_frames, ignore_index=True) if local_frames else pd.DataFrame()
    df_remote = pd.concat(remote_frames, ignore_index=True) if remote_frames else pd.DataFrame()
    merged_df = pd.concat([df_local, df_remote], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset="id", keep="last")
    return merged_df


def process_row(row, index: int, total: int) -> Tuple[str, str, str]:
    logger.info("Processing row %d/%d for lab/lcn/lpn started.", index, total)
    lab_text = process_text(normalize_text_list(row.get("lab", [])))
    lcn_text = process_text(normalize_text_list(row.get("lcn", [])))
    lnp_text = process_text(normalize_text_list(row.get("lpn", [])))
    logger.info("Processing row %d/%d for lab/lcn/lpn completed.", index, total)
    return lab_text, lcn_text, lnp_text


def preprocess_combined(input_frame: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(input_frame)
    combined_rows = []
    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        logger.info("Processing row %d/%d started.", i, total_rows)
        # Process text-based features
        lab_text, lcn_text, lnp_text = process_row(row, i, total_rows)
        title = process_text(row.get("title", []))
        # Retrieve URI fields directly from the row
        curi = row.get("curi", [])
        puri = row.get("puri", [])
        voc = row.get("voc", [])
        tld = row.get("tld", [])
        sparql = row.get('sparql', [])
        combined_rows.append({
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "title": title,
            "lab": lab_text,
            "lcn": lcn_text,
            "lpn": lnp_text,
            "curi": curi,
            "puri": puri,
            "voc": voc,
            "tlds": tld,
            'sparql': sparql,
            'creator': row.get("creator", ""),
            'license': row.get("license", ""),
            'language': find_language(lab_text[:1000])
        })
        logger.info("Processing row %d/%d completed.", i, total_rows)
    combined_df = pd.DataFrame(combined_rows)
    logger.info("Combined processing complete: %d/%d.", total_rows, total_rows)
    return combined_df


def process_void_row(row, index: int, total: int) -> dict:
    logger.info("Processing void row %d/%d started.", index, total)
    dsc_text = process_text(normalize_text_list(row.get("dsc", [])))
    sbj_text = process_text(normalize_text_list(row.get("sbj", [])))
    logger.info("Processing void row %d/%d completed.", index, total)
    return {"sbj": sbj_text, "dsc": dsc_text}


def preprocess_void(input_frame: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(input_frame)
    processed_rows = []
    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_rows.append(process_void_row(row, i, total_rows))
    out_df = pd.DataFrame({
        "id": input_frame["id"] if "id" in input_frame.columns else list(range(len(processed_rows))),
        "sbj": [r["sbj"] for r in processed_rows],
        "dsc": [r["dsc"] for r in processed_rows]
    })
    logger.info("Void processing complete: %d/%d.", total_rows, total_rows)
    return out_df


def combine_with_void(combined_df: pd.DataFrame, void_df: pd.DataFrame) -> pd.DataFrame:
    merged_final = pd.merge(
        combined_df, void_df, on="id", how="outer", suffixes=("", "_dup")
    )

    # Drop duplicate columns that came with the '_dup' suffix.
    dup_cols = [col for col in merged_final.columns if col.endswith("_dup")]
    merged_final.drop(columns=dup_cols, inplace=True)

    merged_final = merged_final.drop_duplicates(subset="id")
    merged_final = merged_final.dropna(subset=["category"])

    logger.info("Final combined processing complete.")
    return merged_final

def combine_with_void_and_lov_data(combined_df: pd.DataFrame, void_df: pd.DataFrame, lov_df: pd.DataFrame) -> pd.DataFrame:
    return combine_with_void(combine_with_void(combined_df, void_df), lov_df)


def process_all_from_input(input_data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, List[Any]]:
    # Convert dict to DataFrame if needed.
    if isinstance(input_data, dict):
        converted = {}
        for key, value in input_data.items():
            if not isinstance(value, list):
                converted[key] = [value]
            else:
                converted[key] = value
        max_len = max(len(v) for v in converted.values())
        padded = {key: value + [None] * (max_len - len(value)) for key, value in converted.items()}
        df = pd.DataFrame(padded)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        raise ValueError("Input data must be either a dict or a pandas DataFrame")

    logger.info("Converted input data to DataFrame with %d rows", len(df))
    combined_df = preprocess_combined(df)
    void_df = preprocess_void(df)
    merged_df = combine_with_void(combined_df, void_df)
    logger.info("Full processing completed")
    return merged_df.to_dict(orient='list')

def process_lov_data_row(row, index: int, total: int) -> dict:
    logger.info("Processing lov row %d/%d started.", index, total)
    tags = row.get("tags", [])
    comments  = process_text(normalize_text_list(row.get("comments", [])))
    logger.info("Processing lov row %d/%d completed.", index, total)
    return {"tags": tags, "comments": comments}

def preprocess_lov_data(input_frame: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(input_frame)
    processed_rows = []
    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_rows.append(process_lov_data_row(row, i, total_rows))
    out_df = pd.DataFrame({
        "id": input_frame["id"] if "id" in input_frame.columns else list(range(len(processed_rows))),
        "tags": [r["tags"] for r in processed_rows],
        "comments": [r["comments"] for r in processed_rows]
    })
    logger.info("Lov processing complete: %d/%d.", total_rows, total_rows)
    return out_df


def main():
    logger.info("Starting preprocessing workflow")

    df = merge_dataset()
    logger.info(f"Merged dataset: {len(df)} rows")

    combined_df = preprocess_combined(df)
    logger.info(f"Combined processing complete: {len(combined_df)} rows")

    void_df = preprocess_void(merge_void_dataset())
    logger.info(f"Void processing complete: {len(void_df)} rows")

    lov_data = preprocess_lov_data(pd.read_json('../data/raw/lov_cloud/voc_cmt.json'))
    logger.info(f"LOV processing complete: {len(lov_data)} rows")

    # Merge the combined and void DataFrames
    final_df = combine_with_void_and_lov_data(combined_df, void_df, lov_df=lov_data)
    logger.info(f"Final merged dataframe has {len(final_df)} rows")

    # Save final output to combined.json in the PROCESSED_DIR.
    output_path = f"{PROCESSED_DIR}/combined.json"
    final_df.to_json(output_path)
    logger.info("Preprocessing workflow completed. Output saved to %s", output_path)


if __name__ == "__main__":
    main()
