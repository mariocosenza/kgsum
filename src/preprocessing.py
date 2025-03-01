import logging
import os
import re
from os import listdir, path
from typing import Dict, Tuple

import pandas as pd
import spacy
from spacy.language import Language
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

for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    if not path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

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
    logger.info("Processing row %d/%d started.", index, total)

    lab_text = process_text(normalize_text_list(row.get("lab", [])))
    lcn_text = process_text(normalize_text_list(row.get("lcn", [])))
    lnp_text = process_text(normalize_text_list(row.get("lpn", [])))

    logger.info("Processing row %d/%d completed.", index, total)
    return lab_text, lcn_text, lnp_text


def preprocess_lab_lcn_lnp(input_frame: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(input_frame)
    processed_lab = []
    processed_lcn = []
    processed_lnp = []

    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        lab_text, lcn_text, lnp_text = process_row(row, i, total_rows)
        processed_lab.append(lab_text)
        processed_lcn.append(lcn_text)
        processed_lnp.append(lnp_text)

    out_df = pd.DataFrame({
        "id": input_frame["id"],
        "category": input_frame["category"],
        "lab": processed_lab,
        "lcn": processed_lcn,
        "lpn": processed_lnp
    })

    out_df.to_json(f"{PROCESSED_DIR}/lab_lcn_lpn.json")
    logger.info("Processing complete: %d/%d", total_rows, total_rows)
    return out_df


def analyze_uri_metadata(row, index: int, total: int) -> Dict[str, any]:
    logger.info("Analyzing URI metadata for row %d/%d started.", index, total)

    result = {
        "id": row.get("id", ""),
        "category": row.get("category", ""),
        "tlds": set()
    }

    result["curi"] = row.get("curi", [])
    result["puri"] = row.get("puri", [])
    result["voc"] = row.get("voc", [])
    result["tld"] = row.get("tld", [])

    logger.info("Analyzing URI metadata for row %d/%d completed.", index, total)
    return result


def preprocess_voc_curi_puri_tld(input_frame: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(input_frame)
    processed_rows = []

    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_rows.append(analyze_uri_metadata(row, i, total_rows))

    processed_df = pd.DataFrame({
        "id": [row["id"] for row in processed_rows],
        "category": [row["category"] for row in processed_rows],
        "curi": [row["curi"] for row in processed_rows],
        "puri": [row["puri"] for row in processed_rows],
        "voc": [row["voc"] for row in processed_rows],
        "tlds": [row["tld"] for row in processed_rows]
    })

    processed_df.to_json(f"{PROCESSED_DIR}/voc_curi_puri_tlds.json")
    logger.info("URI processing complete: %d/%d", total_rows, total_rows)
    return processed_df


def process_void_row(row, index: int, total: int) -> str:
    logger.info("Processing void row %d/%d started.", index, total)
    result = process_text(normalize_text_list(row.get("dsc", [])))
    logger.info("Processing void row %d/%d completed.", index, total)
    return result


def preprocess_void(input_frame: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(input_frame)
    processed_void = []

    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_void.append(process_void_row(row, i, total_rows))

    out_df = pd.DataFrame({
        "id": input_frame["id"] if "id" in input_frame.columns else range(len(processed_void)),
        "void": processed_void
    })

    out_df.to_json(f"{PROCESSED_DIR}/void.json")
    logger.info("Void processing complete: %d/%d", total_rows, total_rows)
    return out_df


def preprocess_voc_tags(input_frame: pd.DataFrame) -> pd.DataFrame:
    frame = input_frame.dropna(subset=["id", "tags", "voc"])
    frame.to_json(f"{PROCESSED_DIR}/voc_tags.json")
    logger.info("Vocabulary tags processing complete.")
    return frame


def main():
    logger.info("Starting preprocessing workflow")

    df = merge_dataset()
    logger.info(f"Merged dataset: {len(df)} rows")

    lab_lcn_lnp_df = preprocess_lab_lcn_lnp(df)
    logger.info(f"Processed labels and names: {len(lab_lcn_lnp_df)} rows")

    uri_df = preprocess_voc_curi_puri_tld(df)
    logger.info(f"Processed URIs: {len(uri_df)} rows")

    void_df = merge_void_dataset()
    if not void_df.empty:
        void_processed_df = preprocess_void(void_df)
        logger.info(f"Processed void dataset: {len(void_processed_df)} rows")

    logger.info("Preprocessing workflow completed")
