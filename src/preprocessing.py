import logging
from os import listdir

import pandas as pd
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a spaCy language detector factory.
@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()

# Load the main spaCy pipeline (for language detection).
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector", last=True)

# Load language-specific pipelines.
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

def process_text(text: str) -> str:
    if not text or len(text) > 100000:
        return ""
    doc = nlp(text)
    lang = doc._.language.get("language", "en")
    chosen_nlp = pipeline_dict.get(lang, fallback_pipeline)
    docs = list(chosen_nlp.pipe([text]))
    return docs[0].text if docs else ""

def merge_dataset() -> pd.DataFrame:
    local_frames = []
    for file in listdir("../data/raw/local"):
        if "local_feature_set" in file:
            local_frames.append(pd.read_json(f"../data/raw/local/{file}"))
    remote_frames = []
    for file in listdir("../data/raw/remote"):
        if "remote_feature_set" in file:
            remote_frames.append(pd.read_json(f"../data/raw/remote/{file}"))

    df_local = pd.concat(local_frames, ignore_index=True) if local_frames else pd.DataFrame()
    df_remote = pd.concat(remote_frames, ignore_index=True) if remote_frames else pd.DataFrame()

    merged_df = pd.concat([df_local, df_remote], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset="id", keep="last")

    return merged_df

def merge_void_dataset():
    """Merge local and remote void datasets."""
    local_frames = []
    for file in listdir("../data/raw/local"):
        if "local_void_feature_set" in file:
            local_frames.append(pd.read_json(f"../data/raw/local/{file}"))
    remote_frames = []
    for file in listdir("../data/raw/remote"):
        if "remote_void_feature_set" in file:
            remote_frames.append(pd.read_json(f"../data/raw/remote/{file}"))

    df_local = pd.concat(local_frames, ignore_index=True) if local_frames else pd.DataFrame()
    df_remote = pd.concat(remote_frames, ignore_index=True) if remote_frames else pd.DataFrame()

    merged_df = pd.concat([df_local, df_remote], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset="id", keep="last")

    return merged_df

def process_row(row, index: int, total: int):
    logger.info("Processing row %d/%d started.", index, total)

    lab_text = process_text(" ".join(word if word is not None else "" for word in row.get("lab", []))) if row.get("lab") else ""
    lcn_text = process_text(" ".join(word if word is not None else "" for word in row.get("lcn", []))) if row.get("lcn") else ""
    lnp_text = process_text(" ".join(word if word is not None else "" for word in row.get("lpn", []))) if row.get("lpn") else ""

    logger.info("Processing row %d/%d completed.", index, total)
    return lab_text, lcn_text, lnp_text

def preprocess_lab_lcn_lnp(input_frame: pd.DataFrame):
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
        "lnp": processed_lnp
    })

    out_df.to_json("../data/processed/lab_lcn_lnp.json")
    logger.info("Processing complete: %d/%d", total_rows, total_rows)

def process_void_row(row, index: int, total: int) -> str:
    logger.info("Processing void row %d/%d started.", index, total)

    result = process_text(" ".join(word if word is not None else "" for word in row.get("dsc", []))) if row.get("dsc") else ""

    logger.info("Processing void row %d/%d completed.", index, total)
    return result

def preprocess_void(input_frame: pd.DataFrame):
    total_rows = len(input_frame)
    processed_void = []

    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_void.append(process_void_row(row, i, total_rows))
    pd.DataFrame({"void": processed_void}).to_json("../data/processed/void.json")

    logger.info("Void processing complete: %d/%d", total_rows, total_rows)

def preprocess_voc_tags(input_frame: pd.DataFrame):
    frame = input_frame.dropna()
    frame.to_json("../data/processed/voc_tags.json")
    logger.info("voc tags processing complete.")

def preprocess_voc_curi_puri_tld(input_frame: pd.DataFrame):
    processed_frame = pd.DataFrame()
    processed_frame.to_json("../data/processed/voc_curi_puri_tld.json")
    logger.info("voc curi puri tld processing complete.")

if __name__ == "__main__":
    df = merge_dataset()
    preprocess_lab_lcn_lnp(df)
