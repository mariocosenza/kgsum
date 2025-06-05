import logging
import os
import re
from os import listdir
from typing import Any

import pandas as pd
import spacy
from spacy import Language
from spacy_langdetect import LanguageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ——— 1) SpaCy + Language‐Detector setup ———

@Language.factory("language_detector")
def get_lang_detector(nlp: Language, name: str) -> LanguageDetector:
    """
    Factory for spaCy’s LanguageDetector (spacy_langdetect).
    """
    return LanguageDetector()


# Load the main (lightweight) pipeline just for language detection
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector", last=True)

# Attempt to load per‐language pipelines. If any model is missing, log an error and skip it.
pipeline_dict: dict[str, Language] = {}
for lang_code, model_name in [
    ("en", "en_core_web_trf"),
    ("it", "it_core_news_lg"),
    ("es", "es_dep_news_trf"),
    ("de", "de_dep_news_trf"),
    ("nl", "nl_core_news_lg"),
    ("fr", "fr_dep_news_trf"),
    ("ru", "ru_core_news_lg"),
    ("zh", "zh_core_web_trf"),
    ("ja", "ja_core_news_trf"),
    ("pt", "pt_core_news_lg"),
]:
    try:
        pipeline_dict[lang_code] = spacy.load(model_name)
        logger.info("Loaded spaCy pipeline for '%s': %s", lang_code, model_name)
    except Exception as e:
        logger.error("Error loading pipeline for '%s' (%s): %s", lang_code, model_name, e)

# Fallback: a lightweight multilingual UD model
try:
    fallback_pipeline: Language = spacy.load("xx_sent_ud_sm")
    logger.info("Loaded fallback multilingual pipeline: xx_sent_ud_sm")
except Exception as e:
    logger.error("Error loading multilingual fallback pipeline: %s", e)
    # If fallback is missing, at least keep English model as fallback
    fallback_pipeline = pipeline_dict.get("en", nlp)


# ——— 2) Directories & regex patterns ———

DATA_DIR = "../data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure output directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# These patterns help split URIs into (namespace, local_name, tld)
URI_FRAGMENT_PATTERN = re.compile(r"[#/](?P<fragment>[^#/]+)$")
URI_NAMESPACE_PATTERN = re.compile(r"^(?P<ns>.*?)[#/][^#/]+$")
TLD_PATTERN = re.compile(r"^(?:https?://)?(?:www\.)?(?P<tld>[^/]+)")


# ——— 3) Utility functions ———

def sanitize_field(value: Any) -> Any:
    """
    If `value` is an empty list, return "".
    If `value` is the string "[]", also return "".
    Otherwise, if it’s a non‐empty list, remove duplicates.
    Everything else: return unchanged.
    """
    if isinstance(value, list) and not value:
        return ""
    if isinstance(value, str) and value.strip() == "[]":
        return ""
    if isinstance(value, list):
        return remove_duplicates(value)
    return value


def analyze_uri(uri: str) -> dict[str, str]:
    """
    Given a URI, extract:
      - namespace  (everything before the final "#" or "/")
      - local_name (the piece after the final "#" or "/")
      - tld        (top‐level domain or host, via TLD_PATTERN)
    If anything is missing or input is not a str, fields default to "".
    """
    result = {"namespace": "", "local_name": "", "tld": ""}
    if not uri or not isinstance(uri, str):
        return result

    # local_name: text after last '#' or '/'
    frag_match = URI_FRAGMENT_PATTERN.search(uri)
    if frag_match:
        result["local_name"] = frag_match.group("fragment")

    # namespace: everything up to (but not including) the last '#' or '/'
    ns_match = URI_NAMESPACE_PATTERN.match(uri)
    if ns_match:
        result["namespace"] = ns_match.group("ns")
    else:
        # if no separator, treat full URI as namespace
        result["namespace"] = uri

    # tld: host or domain from URL
    tld_match = TLD_PATTERN.match(uri)
    if tld_match:
        result["tld"] = tld_match.group("tld")

    return result


def remove_non_ascii(text: Any) -> Any:
    """
    Remove non‐ASCII chars from a string. If not a str, return input unchanged.
    """
    if not isinstance(text, str):
        return text
    return text.encode("ascii", "ignore").decode("ascii")


def process_text(text: Any) -> str:
    """
    1) If the input is not a string (or is too long), return "".
    2) Detect language with the small `en_core_web_sm + language_detector`.
    3) Route the text through the corresponding language pipeline (or fallback).
    4) Finally, strip out any non‐ASCII chars.
    """
    if not isinstance(text, str) or not text or len(text) > 100_000:
        return ""

    try:
        # 1) detect language
        doc = nlp(text)
        lang_code = doc._.language.get("language", "xx")

        # 2) pick pipeline: if missing, fallback
        chosen_nlp = pipeline_dict.get(lang_code, fallback_pipeline)

        # 3) run only tokenization/serialization (faster than re‐parsing)
        parsed = list(chosen_nlp.pipe([text], disable=["ner", "parser", "tagger"]))  # skip heavy components if not needed
        clean_text = parsed[0].text if parsed else ""
    except Exception as exc:
        logger.error("Error in process_text(\"%s\"): %s", text[:50], exc)
        clean_text = text  # fallback to original if spaCy fails

    # 4) remove non‐ASCII
    return remove_non_ascii(clean_text)


def find_language(text: Any) -> str:
    """
    Return the detected language code of `text` via the small `nlp` pipeline.
    If detection fails, default to "xx".
    """
    if not isinstance(text, str) or not text:
        return "xx"

    try:
        doc = nlp(text)
        return doc._.language.get("language", "xx")
    except Exception as exc:
        logger.error("Error in find_language(\"%s\"): %s", text[:50], exc)
        return "xx"


def normalize_text_list(text_list: Any) -> str:
    """
    If `text_list` is a string, return it unchanged.
    If it's a list of tokens/strings, concatenate with spaces.
    Otherwise, return "".
    """
    if not text_list:
        return ""
    if isinstance(text_list, str):
        return text_list
    if isinstance(text_list, list):
        return " ".join(str(x) for x in text_list if x is not None)
    return ""


def process_normalize_text(text_list: Any) -> list[str]:
    """
    Given a list of values, cast each to str and run through `process_text`.
    Non‐list inputs yield an empty list.
    """
    if not isinstance(text_list, list):
        return []
    return [process_text(str(x)) for x in text_list if x is not None]


def extract_named_entities(lab_list: Any) -> list[str]:
    """
    For each element in `lab_list` (expected to be a list of strings),
    detect its language, run the appropriate pipeline, and collect unique entity labels.
    If lab_list is not a list, return [].
    """
    if not isinstance(lab_list, list):
        return []

    entity_types: set[str] = set()
    for text in lab_list:
        if not isinstance(text, str) or not text:
            continue
        try:
            # 1) detect language
            doc0 = nlp(text)
            lang_code = doc0._.language.get("language", "xx")

            # 2) choose pipeline (or fallback)
            chosen_nlp = pipeline_dict.get(lang_code, fallback_pipeline)
            doc1 = chosen_nlp(text)

            # 3) collect entity labels
            for ent in doc1.ents:
                if ent.label_:
                    entity_types.add(ent.label_)
        except Exception as e:
            logger.error("NER failure on \"%s\": %s", text[:50], e)
    return sorted(entity_types)


# ——— 4) I/O / merging functions ———

def merge_dataset() -> pd.DataFrame:
    """
    Read all JSON files under RAW_DIR/local whose names contain "local_feature_set",
    then all JSONs under RAW_DIR/remote reading "remote_feature_set". Concatenate, dedupe by "id".
    """
    local_frames: list[pd.DataFrame] = []
    remote_frames: list[pd.DataFrame] = []

    # 1) load local
    local_path = os.path.join(RAW_DIR, "local")
    for fname in listdir(local_path):
        if "local_feature_set" in fname and fname.endswith(".json"):
            fullpath = os.path.join(local_path, fname)
            try:
                local_frames.append(pd.read_json(fullpath))
            except Exception as exc:
                logger.error("Failed to read %s: %s", fullpath, exc)

    # 2) load remote
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
    """
    Same as merge_dataset but looking for "local_void_feature_set" / "remote_void_feature_set".
    """
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


# ——— 5) Row‐level processing ———

def process_row(row: dict[str, Any], idx: int, total: int) -> tuple[str, str, str]:
    """
    Given a single record (as dict), process fields "lab"/"lcn"/"lpn" by:
      1) normalize list→string
      2) detect language + run pipeline
      3) strip to ASCII
    Returns (lab_text, lcn_text, lpn_text).
    """
    logger.info(" Processing row %d/%d …", idx, total)

    # Normalize, then run process_text
    lab_text = process_text(normalize_text_list(row.get("lab", [])))
    lcn_text = process_text(normalize_text_list(row.get("lcn", [])))
    lpn_text = process_text(normalize_text_list(row.get("lpn", [])))

    logger.info(" Completed row %d/%d.", idx, total)
    return lab_text, lcn_text, lpn_text


def preprocess_combined(input_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build a new DataFrame with the following columns for each row:
      - "id", "category", "title"
      - processed versions of "lab", "lcn", "lpn"
      - sanitized versions of fields: "curi", "puri", "voc", "tlds", "sparql"
      - "creator", "license"
      - "ner" (list of unique entity labels from lab),
      - "language" (detected from lab_text),
      - "con"
    """
    total = len(input_frame)
    combined_rows: list[dict[str, Any]] = []

    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        lab_text, lcn_text, lpn_text = process_row(row, i, total)

        # Title: first sanitize (empty‐list→""), then process_text
        title_raw = sanitize_field(row.get("title", ""))
        title = process_text(title_raw)

        # The following fields may be lists or strings — sanitize them
        curi = sanitize_field(row.get("curi", ""))
        puri = sanitize_field(row.get("puri", ""))
        voc = sanitize_field(row.get("voc", ""))
        tlds = sanitize_field(row.get("tlds", ""))
        sparql = sanitize_field(row.get("sparql", ""))
        creator = row.get("creator", "")
        license_ = row.get("license", "")

        # NER: pass the original list of strings from "lab", not the processed single string
        lab_list = row.get("lab", [])
        if not isinstance(lab_list, list):
            lab_list = [lab_list] if lab_list else []
        ner_types = extract_named_entities(lab_list)

        # language: detect on the (possibly long) lab_text; just take first 1000 chars
        language = find_language(lab_text[:1000])

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


def process_void_row(row: dict[str, Any], idx: int, total: int) -> dict[str, str]:
    """
    For a 'void' record, just process fields 'dsc' and 'sbj' the same way as above.
    """
    logger.info(" Processing void row %d/%d …", idx, total)
    dsc_text = process_text(normalize_text_list(row.get("dsc", [])))
    sbj_text = process_text(normalize_text_list(row.get("sbj", [])))
    logger.info(" Completed void row %d/%d.", idx, total)
    return {"sbj": sbj_text, "dsc": dsc_text}


def preprocess_void(input_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build a two‐column DataFrame from `input_frame`:
      - "id"
      - "sbj", "dsc" (both processed & ASCII‐cleaned)
    """
    total = len(input_frame)
    processed_rows: list[dict[str, str]] = []

    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_rows.append(process_void_row(row, i, total))

    out_df = pd.DataFrame({
        "id": input_frame["id"] if "id" in input_frame.columns else list(range(total)),
        "sbj": [r["sbj"] for r in processed_rows],
        "dsc": [r["dsc"] for r in processed_rows],
    })
    logger.info("Void processing complete: %d/%d.", len(out_df), total)
    return out_df


def combine_with_void(combined_df: pd.DataFrame, void_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left‐outer merge of combined_df with void_df on "id", then:
      - drop any "_dup" columns that came from the right side,
      - drop duplicate ids,
      - drop rows with no 'category'.
    """
    merged_final = pd.merge(
        combined_df, void_df, on="id", how="outer", suffixes=("", "_dup")
    )

    # Drop duplicate columns that end with "_dup"
    dup_cols = [col for col in merged_final.columns if col.endswith("_dup")]
    if dup_cols:
        merged_final.drop(columns=dup_cols, inplace=True)

    # Drop possible duplicate 'id' rows
    if "id" in merged_final.columns:
        merged_final = merged_final.drop_duplicates(subset="id")

    # Drop any rows where 'category' is missing or empty
    if "category" in merged_final.columns:
        merged_final = merged_final.dropna(subset=["category"])
        merged_final = merged_final[merged_final["category"].astype(str).ne("")]

    logger.info("Merged with void; resulting rows: %d", len(merged_final))
    return merged_final


def combine_with_void_and_lov_data(
    combined_df: pd.DataFrame, void_df: pd.DataFrame, lov_df: pd.DataFrame
) -> pd.DataFrame:
    """
    First merge combined_df + void_df, then merge that result with lov_df, both on 'id'.
    Finally, drop any duplicate columns and ensure 'category' is present.
    """
    temp = combine_with_void(combined_df, void_df)
    final = combine_with_void(temp, lov_df)
    return final


def remove_empty_list_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cell in `df`, if it’s an empty list or the string "[]", replace with "".
    """
    def _replacer(x: Any) -> Any:
        if isinstance(x, list) and not x:
            return ""
        if isinstance(x, str) and x.strip() == "[]":
            return ""
        return x

    return df.map(_replacer)


def remove_duplicates(series_or_list: Any) -> list[str]:
    """
    Given a pandas Series or Python list, return a deduplicated list of strings,
    dropping None, 'None', or "" if they appear.
    """
    if isinstance(series_or_list, pd.Series):
        items = series_or_list.dropna().astype(str).tolist()
    elif isinstance(series_or_list, list):
        items = [str(x) for x in series_or_list]
    else:
        return []

    unique = set(items)
    # Clean out any literal "None" or empty string
    unique.discard("None")
    unique.discard("")
    return sorted(unique)


# ——— 6) LOV data processing ———

def process_lov_data_row(row: dict[str, Any], idx: int, total: int) -> dict[str, Any]:
    """
    For each LOV row (expects columns 'tags' (list) and 'comments' (list of strings)),
    sanitize tags, then process each comment through process_normalize_text.
    """
    logger.info(" Processing LOV row %d/%d …", idx, total)
    tags = sanitize_field(row.get("tags", []))

    comments = row.get("comments", None)
    if isinstance(comments, list):
        comments = process_normalize_text(comments)
    else:
        comments = []

    logger.info(" Completed LOV row %d/%d.", idx, total)
    return {"tags": tags, "comments": comments}


def preprocess_lov_data(input_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame with columns:
      - "id"
      - "tags"   (sanitized & deduped)
      - "comments" (each comment processed & ASCII‐cleaned)
    """
    total = len(input_frame)
    processed_rows: list[dict[str, Any]] = []

    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_rows.append(process_lov_data_row(row, i, total))

    out_df = pd.DataFrame({
        "id": input_frame["id"] if "id" in input_frame.columns else list(range(total)),
        "tags": [r["tags"] for r in processed_rows],
        "comments": [r["comments"] for r in processed_rows],
    })
    logger.info("LOV processing complete: %d/%d.", len(out_df), total)
    return out_df


# ——— 7) End‐to‐end helpers ———

def process_all_from_input(input_data: Any) -> dict[str, list[Any]]:
    """
    Accept either:
      - a dict[str, Any], where each value is either a list or a scalar (convert to DataFrame),
      - or a pandas DataFrame directly.

    Returns a single dict of aggregated lists:
      'id', 'title', 'lab', 'lcn', 'lpn', 'curi', 'puri', 'voc', 'tlds', 'sparql',
      'creator', 'license', 'language', 'dsc', 'sbj', 'ner', 'con'
    """
    if isinstance(input_data, dict):
        # Ensure each key maps to a list, then pad all lists to same length
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
    combined_df = preprocess_combined(df)
    void_df = preprocess_void(df)

    # Build final output as a dict of lists:
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
        "ner": remove_duplicates(sum(combined_df["ner"].tolist(), [])),  # flatten list of lists
        "con": remove_duplicates(combined_df["con"].tolist()),
    }


def main() -> None:
    logger.info("Starting preprocessing workflow.")

    df = merge_dataset()
    logger.info("Merged dataset contains %d rows.", len(df))

    combined_df = preprocess_combined(df)
    logger.info("After combined preprocessing: %d rows.", len(combined_df))

    void_df = preprocess_void(merge_void_dataset())
    logger.info("After void preprocessing: %d rows.", len(void_df))

    lov_raw = pd.read_json(os.path.join(RAW_DIR, "lov_cloud", "voc_cmt.json"))
    logger.info("Loaded LOV raw data: %d rows.", len(lov_raw))

    lov_data = preprocess_lov_data(lov_raw)
    logger.info("After LOV preprocessing: %d rows.", len(lov_data))

    final_df = combine_with_void_and_lov_data(combined_df, void_df, lov_data)
    final_df = remove_empty_list_values(final_df)
    logger.info("Final merged DataFrame: %d rows.", len(final_df))

    output_path = os.path.join(PROCESSED_DIR, "combined.json")
    final_df.to_json(output_path, orient="records", lines=False)
    logger.info("Preprocessing complete. Saved to %s", output_path)


if __name__ == "__main__":
    main()
