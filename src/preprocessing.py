import logging
import argparse
import os
import re
from os import listdir
from typing import Any

import pandas as pd
import spacy

# --- language detection: use langdetect ---
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1) SpaCy setup (full pipeline) with GPU option ---

SPACY_LANGS = {
    "en": "en_core_web_trf",
    "it": "it_core_news_lg",
    "es": "es_dep_news_trf",
    "de": "de_dep_news_trf",
    "nl": "nl_core_news_lg",
    "fr": "fr_dep_news_trf",
    "ru": "ru_core_news_lg",
    "zh": "zh_core_web_trf",
    "ja": "ja_core_news_trf",
    "pt": "pt_core_news_lg",
}

USE_GPU_ON_IMPORT = False  # Set True to always use GPU at import

def _load_spacy_pipelines_on_demand(use_gpu: bool = False):
    """Returns an empty dict, pipelines will be loaded on demand."""
    if use_gpu:
        try:
            spacy.require_gpu()
            logger.info("spaCy using GPU for pipeline(s)")
        except Exception as e:
            logger.warning("Could not enable spaCy GPU mode: %s", e)
    return {}, None  # Pipelines will be loaded only when needed

pipeline_dict, fallback_pipeline = _load_spacy_pipelines_on_demand(use_gpu=USE_GPU_ON_IMPORT)

def get_or_load_pipeline(lang_code: str, pipeline_dict_local=None, fallback_pipeline_local=None):
    """Loads and returns the spacy pipeline for a given lang_code, loading on demand."""
    use_dict = pipeline_dict_local if pipeline_dict_local is not None else pipeline_dict
    use_fallback = fallback_pipeline_local if fallback_pipeline_local is not None else fallback_pipeline

    if lang_code in use_dict:
        return use_dict[lang_code]
    if lang_code in SPACY_LANGS:
        model_name = SPACY_LANGS[lang_code]
        try:
            nlp = spacy.load(model_name)
            use_dict[lang_code] = nlp
            logger.info("Loaded spaCy pipeline for '%s': %s", lang_code, model_name)
            return nlp
        except Exception as e:
            logger.warning("SpaCy pipeline missing for '%s' (%s): %s", lang_code, model_name, e)
    # Fallback: try to load multilingual, or fallback to English blank
    if "xx" not in use_dict:
        try:
            use_dict["xx"] = spacy.load("xx_sent_ud_sm")
            logger.info("Loaded fallback multilingual pipeline: xx_sent_ud_sm")
        except Exception as e:
            logger.warning("Error loading multilingual fallback pipeline: %s", e)
            use_dict["xx"] = spacy.blank("en")
    return use_dict["xx"]

def setup_spacy_pipelines(use_gpu: bool = False):
    global pipeline_dict, fallback_pipeline
    return pipeline_dict, fallback_pipeline

def get_spacy_lang_code(detected: str, pipeline_dict_local=None) -> str:
    """Return spaCy-supported language code if available, else 'xx'."""
    # We now only load on demand, so just check if supported
    return detected if detected in SPACY_LANGS else "xx"

# --- 2) Language Detection via langdetect ---

def find_language(text: Any, pipeline_dict_local=None) -> str:
    """
    Return the detected language code of `text` using langdetect.
    If detection fails, default to "xx".
    """
    if not isinstance(text, str) or not text:
        return "xx"
    try:
        code = detect(text)
        return get_spacy_lang_code(code)
    except LangDetectException:
        return "xx"
    except Exception as exc:
        logger.error("Error in find_language(\"%s\"): %s", str(text)[:50], exc)
        return "xx"

# --- 3) spaCy batch normalization for selected features ---

def spacy_clean_normalize_batch(texts, pipeline_dict_local=None, fallback_pipeline_local=None, batch_size=32, n_process=1):
    """
    Processes a list of texts using spaCy's full pipeline in batches, with multiprocessing.
    Returns a list of normalized strings.
    """
    use_dict = pipeline_dict_local if pipeline_dict_local is not None else pipeline_dict
    use_fallback = fallback_pipeline_local if fallback_pipeline_local is not None else fallback_pipeline
    if not isinstance(texts, list) or not texts:
        return []
    # Detect languages for each text
    langs = [find_language(text, use_dict) if isinstance(text, str) and text else "xx" for text in texts]
    # Group texts by language for efficient batching
    from collections import defaultdict
    lang2idxs = defaultdict(list)
    for idx, lang in enumerate(langs):
        lang2idxs[lang].append(idx)
    result = ["" for _ in texts]
    for lang, idxs in lang2idxs.items():
        nlp = get_or_load_pipeline(lang, use_dict, use_fallback)
        sub_texts = [texts[i] for i in idxs]
        for doc, i in zip(nlp.pipe(sub_texts, batch_size=batch_size, n_process=n_process), idxs):
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
            ]
            result[i] = " ".join(tokens)
    return result

def spacy_clean_normalize(text, pipeline_dict_local=None, fallback_pipeline_local=None, batch_size=32, n_process=1):
    """Single string normalize (for string fields), using batch for uniformity."""
    if not isinstance(text, str) or not text:
        return ""
    return spacy_clean_normalize_batch([text], pipeline_dict_local, fallback_pipeline_local, batch_size, n_process)[0]

# --- 4) Utility functions ---

def sanitize_field(value: Any) -> Any:
    if isinstance(value, list) and not value:
        return ""
    if isinstance(value, str) and value.strip() == "[]":
        return ""
    if isinstance(value, list):
        return remove_duplicates(value)
    return value

def analyze_uri(uri: str) -> dict[str, str]:
    URI_FRAGMENT_PATTERN = re.compile(r"[#/](?P<fragment>[^#/]+)$")
    URI_NAMESPACE_PATTERN = re.compile(r"^(?P<ns>.*?)[#/][^#/]+$")
    TLD_PATTERN = re.compile(r"^(?:https?://)?(?:www\.)?(?P<tld>[^/]+)")
    result = {"namespace": "", "local_name": "", "tld": ""}
    if not uri or not isinstance(uri, str):
        return result
    frag_match = URI_FRAGMENT_PATTERN.search(uri)
    if frag_match:
        result["local_name"] = frag_match.group("fragment")
    ns_match = URI_NAMESPACE_PATTERN.match(uri)
    if ns_match:
        result["namespace"] = ns_match.group("ns")
    else:
        result["namespace"] = uri
    tld_match = TLD_PATTERN.match(uri)
    if tld_match:
        result["tld"] = tld_match.group("tld")
    return result

def normalize_text_list(text_list: Any) -> str:
    if not text_list:
        return ""
    if isinstance(text_list, str):
        return text_list
    if isinstance(text_list, list):
        return " ".join(str(x) for x in text_list if x is not None)
    return ""

def remove_duplicates(series_or_list: Any) -> list[str]:
    if isinstance(series_or_list, pd.Series):
        items = series_or_list.dropna().astype(str).tolist()
    elif isinstance(series_or_list, list):
        items = [str(x) for x in series_or_list]
    else:
        return []
    unique = set(items)
    unique.discard("None")
    unique.discard("")
    return sorted(unique)

def remove_empty_list_values(df: pd.DataFrame) -> pd.DataFrame:
    def _replacer(x: Any) -> Any:
        if isinstance(x, list) and not x:
            return ""
        if isinstance(x, str) and x.strip() == "[]":
            return ""
        return x
    return df.map(_replacer)

# --- 5) NER extraction for lab (optional) ---

def extract_named_entities(lab_list: Any, pipeline_dict_local=None, fallback_pipeline_local=None, use_ner: bool = True) -> list[str]:
    """
    For each string in lab_list, detect its language, use the appropriate spaCy pipeline
    (falling back to xx if language is not supported), and collect unique entity labels.
    If use_ner is False, return [].
    """
    if not use_ner or not isinstance(lab_list, list):
        return []
    use_dict = pipeline_dict_local if pipeline_dict_local is not None else pipeline_dict
    use_fallback = fallback_pipeline_local if fallback_pipeline_local is not None else fallback_pipeline
    entity_types: set[str] = set()
    for text in lab_list:
        if not isinstance(text, str) or not text:
            continue
        lang_code = find_language(text, use_dict)
        chosen_nlp = get_or_load_pipeline(lang_code, use_dict, use_fallback)
        try:
            doc = chosen_nlp(text)
            for ent in doc.ents:
                if ent.label_:
                    entity_types.add(ent.label_)
        except Exception as e:
            logger.error("NER failure on \"%s\": %s", text[:50], e)
    return sorted(entity_types)

# --- 6) I/O / merging functions ---

DATA_DIR = "../data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def merge_dataset() -> pd.DataFrame:
    local_frames: list[pd.DataFrame] = []
    remote_frames: list[pd.DataFrame] = []
    local_path = os.path.join(RAW_DIR, "local")
    for fname in listdir(local_path):
        if "local_feature_set" in fname and fname.endswith(".json"):
            fullpath = os.path.join(local_path, fname)
            try:
                local_frames.append(pd.read_json(fullpath))
            except Exception as exc:
                logger.error("Failed to read %s: %s", fullpath, exc)
    remote_path = os.path.join(RAW_DIR, "remote")
    for fname in listdir(remote_path):
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
    for fname in listdir(local_path):
        if "local_void_feature_set" in fname and fname.endswith(".json"):
            fullpath = os.path.join(local_path, fname)
            try:
                local_frames.append(pd.read_json(fullpath))
            except Exception as exc:
                logger.error("Failed to read void file %s: %s", fullpath, exc)
    remote_path = os.path.join(RAW_DIR, "remote")
    for fname in listdir(remote_path):
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

# --- 7) Row‐level processing ---

def process_row(row: dict[str, Any], idx: int, total: int, pipeline_dict=None, fallback_pipeline=None, batch_size=32, n_process=1) -> tuple[str, str, str]:
    logger.info(" Processing row %d/%d …", idx, total)
    # Normalize lab (list), lcn (string), lpn (string)
    lab_batch_normalized = spacy_clean_normalize_batch(row.get("lab", []), pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process)
    lab_text = " ".join(lab_batch_normalized)
    lcn_text = spacy_clean_normalize(normalize_text_list(row.get("lcn", [])), pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process)
    lpn_text = spacy_clean_normalize(normalize_text_list(row.get("lpn", [])), pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process)
    logger.info(" Completed row %d/%d.", idx, total)
    return lab_text, lcn_text, lpn_text

def preprocess_combined(input_frame: pd.DataFrame, pipeline_dict, fallback_pipeline, use_ner: bool = True, batch_size=32, n_process=1) -> pd.DataFrame:
    total = len(input_frame)
    combined_rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        lab_text, lcn_text, lpn_text = process_row(row, i, total, pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process)
        title_raw = sanitize_field(row.get("title", ""))
        title = title_raw  # No normalization for title
        curi = sanitize_field(row.get("curi", ""))
        puri = sanitize_field(row.get("puri", ""))
        voc = sanitize_field(row.get("voc", ""))
        tlds = sanitize_field(row.get("tlds", ""))
        sparql = sanitize_field(row.get("sparql", ""))
        creator = row.get("creator", "")
        license_ = row.get("license", "")
        lab_list = row.get("lab", [])
        if not isinstance(lab_list, list):
            lab_list = [lab_list] if lab_list else []
        ner_types = extract_named_entities(lab_list, pipeline_dict, fallback_pipeline, use_ner=use_ner)
        language = find_language(lab_text[:1000], pipeline_dict)
        combined_rows.append({
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "title": title,
            "lab": lab_text,
            "lcn": lcn_text,
            "lpn": lpn_text,
            "curi": curi,
            "puri": puri,
            "voc": voc,
            "tlds": tlds,
            "sparql": sparql,
            "creator": creator,
            "license": license_,
            "ner": ner_types,
            "language": language,
            "con": row.get("con", "")
        })
    combined_df = pd.DataFrame(combined_rows)
    logger.info("Combined processing complete: %d/%d.", len(combined_df), total)
    return combined_df

def process_void_row(row: dict[str, Any], idx: int, total: int, pipeline_dict=None, fallback_pipeline=None, batch_size=32, n_process=1) -> dict[str, str]:
    logger.info(" Processing void row %d/%d …", idx, total)
    dsc_text = spacy_clean_normalize(normalize_text_list(row.get("dsc", [])), pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process)
    sbj_text = spacy_clean_normalize(normalize_text_list(row.get("sbj", [])), pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process)
    logger.info(" Completed void row %d/%d.", idx, total)
    return {"sbj": sbj_text, "dsc": dsc_text}

def preprocess_void(input_frame: pd.DataFrame, pipeline_dict=None, fallback_pipeline=None, batch_size=32, n_process=1) -> pd.DataFrame:
    total = len(input_frame)
    processed_rows: list[dict[str, str]] = []
    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_rows.append(process_void_row(row, i, total, pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process))
    out_df = pd.DataFrame({
        "id": input_frame["id"] if "id" in input_frame.columns else list(range(total)),
        "sbj": [r["sbj"] for r in processed_rows],
        "dsc": [r["dsc"] for r in processed_rows],
    })
    logger.info("Void processing complete: %d/%d.", len(out_df), total)
    return out_df

def combine_with_void(combined_df: pd.DataFrame, void_df: pd.DataFrame) -> pd.DataFrame:
    merged_final = pd.merge(
        combined_df, void_df, on="id", how="outer", suffixes=("", "_dup")
    )
    dup_cols = [col for col in merged_final.columns if col.endswith("_dup")]
    if dup_cols:
       merged_final = merged_final.drop(columns=dup_cols)
    if "id" in merged_final.columns:
        merged_final = merged_final.drop_duplicates(subset="id")
    if "category" in merged_final.columns:
        merged_final = merged_final.dropna(subset=["category"])
        merged_final = merged_final[merged_final["category"].astype(str).ne("")]
    logger.info("Merged with void; resulting rows: %d", len(merged_final))
    return merged_final

def combine_with_void_and_lov_data(
    combined_df: pd.DataFrame, void_df: pd.DataFrame, lov_df: pd.DataFrame
) -> pd.DataFrame:
    temp = combine_with_void(combined_df, void_df)
    final = combine_with_void(temp, lov_df)
    return final

# --- 8) LOV data processing ---

def process_lov_data_row(row: dict[str, Any], idx: int, total: int, pipeline_dict=None, fallback_pipeline=None, batch_size=32, n_process=1) -> dict[str, Any]:
    logger.info(" Processing LOV row %d/%d …", idx, total)
    tags = sanitize_field(row.get("tags", []))
    # Flatten comments field before batch processing
    def _flatten(l):
        for el in l:
            if isinstance(el, list):
                yield from _flatten(el)
            else:
                yield el
    comments_value = row.get("comments", [])
    if isinstance(comments_value, list):
        comments_list = list(_flatten(comments_value))
    else:
        comments_list = [comments_value]
    comments_batch_normalized = spacy_clean_normalize_batch(comments_list, pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process)
    comments = " ".join(comments_batch_normalized)
    logger.info(" Completed LOV row %d/%d.", idx, total)
    return {"tags": tags, "comments": comments}

def preprocess_lov_data(input_frame: pd.DataFrame, pipeline_dict=None, fallback_pipeline=None, batch_size=32, n_process=1) -> pd.DataFrame:
    total = len(input_frame)
    processed_rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_rows.append(process_lov_data_row(row, i, total, pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process))
    out_df = pd.DataFrame({
        "id": input_frame["id"] if "id" in input_frame.columns else list(range(total)),
        "tags": [r["tags"] for r in processed_rows],
        "comments": [r["comments"] for r in processed_rows],
    })
    logger.info("LOV processing complete: %d/%d.", len(out_df), total)
    return out_df

# --- 9) End‐to‐end helpers ---

def process_all_from_input(
    input_data: Any,
    use_ner: bool = True,
    batch_size: int = 32,
    n_process: int = 1,
    use_gpu: bool = False,
) -> dict[str, list[Any]]:
    """
    Self-contained: sets up spaCy pipelines internally and processes the input.
    """
    # NOTE: now uses globally loaded pipeline_dict and fallback_pipeline
    if isinstance(input_data, dict):
        converted: dict[str, list[Any]] = {}
        for k, v in input_data.items():
            converted[k] = v if isinstance(v, list) else [v]
        max_len = max(len(lst) for lst in converted.values())
        for k, lst in converted.items():
            if len(lst) < max_len:
                converted[k] = lst + [None] * (max_len - len(lst))
        df = pd.DataFrame(converted)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("Input must be a dict or a pandas DataFrame")
    logger.info("Converted input data to DataFrame (%d rows).", len(df))
    combined_df = preprocess_combined(
        df, pipeline_dict, fallback_pipeline,
        use_ner=use_ner, batch_size=batch_size, n_process=n_process
    )
    void_df = preprocess_void(
        df, pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process
    )
    return {
        "id": remove_duplicates(combined_df["id"].tolist()),
        "title": remove_duplicates(combined_df["title"].tolist()),
        "lab": remove_duplicates(combined_df["lab"].tolist()),
        "lcn": remove_duplicates(combined_df["lcn"].tolist()),
        "lpn": remove_duplicates(combined_df["lpn"].tolist()),
        "curi": remove_duplicates(combined_df["curi"].tolist()),
        "puri": remove_duplicates(combined_df["puri"].tolist()),
        "voc": remove_duplicates(combined_df["voc"].tolist()),
        "tlds": remove_duplicates(combined_df["tlds"].tolist()),
        "sparql": remove_duplicates(combined_df["sparql"].tolist()),
        "creator": remove_duplicates(combined_df["creator"].tolist()),
        "license": remove_duplicates(combined_df["license"].tolist()),
        "language": remove_duplicates(combined_df["language"].tolist()),
        "dsc": remove_duplicates(void_df["dsc"].tolist()),
        "sbj": remove_duplicates(void_df["sbj"].tolist()),
        "ner": remove_duplicates(sum(combined_df["ner"].tolist(), [])),
        "con": remove_duplicates(combined_df["con"].tolist()),
    }

def main(use_ner: bool = True, use_gpu: bool = False, batch_size: int = 32, n_process: int = 1) -> None:
    logger.info("Starting preprocessing workflow. NER enabled: %s, GPU enabled: %s, n_process: %d", use_ner, use_gpu, n_process)
    # NOTE: uses globally loaded pipeline_dict and fallback_pipeline, pipelines are loaded on demand
    df = merge_dataset()
    logger.info("Merged dataset contains %d rows.", len(df))
    combined_df = preprocess_combined(df, pipeline_dict, fallback_pipeline, use_ner=use_ner, batch_size=batch_size, n_process=n_process)
    logger.info("After combined preprocessing: %d rows.", len(combined_df))
    void_df = preprocess_void(merge_void_dataset(), pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process)
    logger.info("After void preprocessing: %d rows.", len(void_df))
    lov_raw = pd.read_json(os.path.join(RAW_DIR, "lov_cloud", "voc_cmt.json"))
    logger.info("Loaded LOV raw data: %d rows.", len(lov_raw))
    lov_data = preprocess_lov_data(lov_raw, pipeline_dict, fallback_pipeline, batch_size=batch_size, n_process=n_process)
    logger.info("After LOV preprocessing: %d rows.", len(lov_data))
    final_df = combine_with_void_and_lov_data(combined_df, void_df, lov_data)
    final_df = remove_empty_list_values(final_df)
    logger.info("Final merged DataFrame: %d rows.", len(final_df))
    output_path = os.path.join(PROCESSED_DIR, "combined.json")
    final_df.to_json(output_path, orient="records", lines=False)
    logger.info("Preprocessing complete. Saved to %s", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset with optional NER, GPU, and parallelism.")
    parser.add_argument("--no-ner", type=bool, default=False, help="Disable NER and set ner field to [].")
    parser.add_argument("--gpu", type=bool, default=False, help="Enable GPU for spaCy pipelines if available.")
    parser.add_argument("--batch-size", type=int, default=32, help="spaCy batch size.")
    parser.add_argument("--n-process", type=int, default=1, help="Number of processes for spaCy batching (CPU only).")
    args = parser.parse_args()
    main(use_ner=not args.no_ner, use_gpu=args.gpu, batch_size=args.batch_size, n_process=args.n_process)