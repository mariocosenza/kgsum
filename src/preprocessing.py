import argparse
import logging
import os
import re
from typing import Any

import pandas as pd
import spacy
from langdetect import detect, DetectorFactory, LangDetectException
from pandas import Series

from config import Config
from src.util import is_curi_allowed, is_voc_allowed, merge_dataset, merge_void_dataset, \
    RAW_DIR, PROCESSED_DIR

DetectorFactory.seed = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

USE_GPU_ON_IMPORT = False


def _load_spacy_pipelines_on_demand(enable_gpu: bool = False):
    if enable_gpu:
        try:
            spacy.require_gpu()
            logger.info("spaCy using GPU for pipeline(s)")
        except Exception as exc:
            logger.warning("Could not enable spaCy GPU mode: %s", exc)
    return {}, None


pipeline_dict, fallback_pipeline = _load_spacy_pipelines_on_demand(enable_gpu=USE_GPU_ON_IMPORT)


def get_or_load_pipeline(lang_code: str, pipeline_dict_local=None, fallback_pipeline_local=None):
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
        except Exception as exc:
            logger.warning("SpaCy pipeline missing for '%s' (%s): %s", lang_code, model_name, exc)
    # Use fallback if available, otherwise set to blank
    if "xx" not in use_dict:
        if use_fallback is not None:
            use_dict["xx"] = use_fallback
            logger.info("Loaded fallback pipeline from provided fallback_pipeline.")
        else:
            try:
                use_dict["xx"] = spacy.load("xx_sent_ud_sm")
                logger.info("Loaded fallback multilingual pipeline: xx_sent_ud_sm")
            except Exception as exc:
                logger.warning("Error loading multilingual fallback pipeline: %s", exc)
                use_dict["xx"] = spacy.blank("en")
    return use_dict["xx"]


def setup_spacy_pipelines():
    global pipeline_dict, fallback_pipeline
    return pipeline_dict, fallback_pipeline


def get_spacy_lang_code(detected: str) -> str:
    return detected if detected in SPACY_LANGS else "xx"


def find_language(text: Any) -> str:
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


def spacy_clean_normalize_single(text, pipeline_dict_local=None, fallback_pipeline_local=None):
    """Process a single text document with spaCy - no batching"""
    if not isinstance(text, str) or not text.strip():
        return ""

    use_dict = pipeline_dict_local if pipeline_dict_local is not None else pipeline_dict
    use_fallback = fallback_pipeline_local if fallback_pipeline_local is not None else fallback_pipeline

    try:
        lang_code = find_language(text)
        nlp = get_or_load_pipeline(lang_code, use_dict, use_fallback)
        doc = nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        return " ".join(tokens)
    except Exception as exc:
        logger.error("Error processing text: %s", exc)
        return ""


def spacy_clean_normalize_list(texts, pipeline_dict_local=None, fallback_pipeline_local=None):
    """Process a list of texts one by one - no batching"""
    if not isinstance(texts, list) or not texts:
        return []

    result = []
    for text in texts:
        normalized = spacy_clean_normalize_single(text, pipeline_dict_local, fallback_pipeline_local)
        result.append(normalized)
    return result


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


def extract_named_entities(lab_list: Any, pipeline_dict_local=None, fallback_pipeline_local=None,
                           use_ner: bool = True) -> list[str]:
    if not use_ner or not isinstance(lab_list, list):
        return []

    use_dict = pipeline_dict_local if pipeline_dict_local is not None else pipeline_dict
    use_fallback = fallback_pipeline_local if fallback_pipeline_local is not None else fallback_pipeline
    entity_types: set[str] = set()

    for text in lab_list:
        if not isinstance(text, str) or not text:
            continue
        try:
            lang_code = find_language(text)
            chosen_nlp = get_or_load_pipeline(lang_code, use_dict, use_fallback)
            doc = chosen_nlp(text)
            for ent in doc.ents:
                if ent.label_:
                    entity_types.add(ent.label_)
        except Exception as exc:
            logger.error("NER failure on \"%s\": %s", text[:50], exc)
    return sorted(entity_types)


def filter_uri_list(uri_list, filter_func=None):
    """Filter a list of URIs by a provided filter function."""
    if not isinstance(uri_list, list):
        uri_list = [uri_list] if isinstance(uri_list, str) and uri_list else []
    if filter_func is not None:
        return [uri for uri in uri_list if filter_func(uri)]
    return uri_list


def extract_local_names(uri_list):
    """Extract local names (after # or /) from a list of URIs."""
    local_names = set()
    for uri in uri_list:
        if not uri or not isinstance(uri, str):
            continue
        if "#" in uri:
            local_name = uri.split("#")[-1]
        elif "/" in uri:
            local_name = uri.rstrip("/").split("/")[-1]
        else:
            local_name = uri
        if local_name:
            local_names.add(local_name)
    return sorted(local_names)


def process_row(
        row: dict[str, Any] | Series,
        idx: int,
        total: int,
        pipeline_dict_int=None,
        fallback_pipeline_int=None,
        enable_filter: bool = Config.USE_FILTER
) -> tuple[str, list[str], list[str], list[str], list[str], list[str]]:
    logger.info(" Processing row %d/%d …", idx, total)

    # Process lab field - handle as list and process each item individually
    lab_raw = row.get("lab", [])
    if not isinstance(lab_raw, list):
        lab_raw = [lab_raw] if lab_raw else []

    # Process each lab item individually
    lab_normalized = []
    for lab_item in lab_raw:
        if isinstance(lab_item, str) and lab_item.strip():
            normalized = spacy_clean_normalize_single(lab_item, pipeline_dict_int, fallback_pipeline_int)
            if normalized:
                lab_normalized.append(normalized)

    lab_text = " ".join(lab_normalized)

    # Filter and extract lcn/lpn here
    if enable_filter:
        curi = filter_uri_list(sanitize_field(row.get("curi", [])), is_curi_allowed)
        puri = filter_uri_list(sanitize_field(row.get("puri", [])), is_voc_allowed)
        voc = filter_uri_list(sanitize_field(row.get("voc", [])), is_voc_allowed)
    else:
        curi = sanitize_field(row.get("curi", []))
        puri = sanitize_field(row.get("puri", []))
        voc = sanitize_field(row.get("voc", []))

    # Local names
    lcn = extract_local_names(curi)
    lpn = extract_local_names(puri)

    logger.info(" Completed row %d/%d.", idx, total)
    return lab_text, lcn, lpn, curi, puri, voc


def preprocess_combined(
        input_frame: pd.DataFrame,
        pipeline_dict_int,
        fallback_pipeline_int,
        use_ner: bool = True,
        enable_filter: bool = Config.USE_FILTER
) -> pd.DataFrame:
    total = len(input_frame)
    combined_rows: list[dict[str, Any]] = []

    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        lab_text, lcn, lpn, curi, puri, voc = process_row(
            row, i, total, pipeline_dict_int, fallback_pipeline_int, enable_filter=enable_filter
        )

        title_raw = sanitize_field(row.get("title", ""))
        title = title_raw
        tlds = sanitize_field(row.get("tlds", ""))
        sparql = sanitize_field(row.get("sparql", ""))
        creator = row.get("creator", "")
        license_ = row.get("license", "")

        lab_list = row.get("lab", [])
        if not isinstance(lab_list, list):
            lab_list = [lab_list] if lab_list else []

        ner_types = extract_named_entities(lab_list, pipeline_dict_int, fallback_pipeline_int, use_ner=use_ner)
        language = find_language(lab_text[:1000])

        combined_rows.append({
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "title": title,
            "lab": lab_text,
            "lcn": lcn,
            "lpn": lpn,
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


def process_void_row(row: dict[str, Any] | Series, idx: int, total: int, pipeline_dict_int=None,
                     fallback_pipeline_int=None) -> dict[str, str]:
    logger.info(" Processing void row %d/%d …", idx, total)

    dsc_raw = normalize_text_list(row.get("dsc", []))
    dsc_text = spacy_clean_normalize_single(dsc_raw, pipeline_dict_int, fallback_pipeline_int)

    sbj_raw = normalize_text_list(row.get("sbj", []))
    sbj_text = spacy_clean_normalize_single(sbj_raw, pipeline_dict_int, fallback_pipeline_int)

    logger.info(" Completed void row %d/%d.", idx, total)
    return {"sbj": sbj_text, "dsc": dsc_text}


def preprocess_void(input_frame: pd.DataFrame, pipeline_dict_int=None, fallback_pipeline_int=None) -> pd.DataFrame:
    total = len(input_frame)
    processed_rows: list[dict[str, str]] = []

    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_rows.append(process_void_row(row, i, total, pipeline_dict_int, fallback_pipeline_int))

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


def process_lov_data_row(row: dict[str, Any] | Series, idx: int, total: int, pipeline_dict_int=None,
                         fallback_pipeline_int=None) -> dict[str, Any]:
    logger.info(" Processing LOV row %d/%d …", idx, total)
    tags = sanitize_field(row.get("tags", []))

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

    # Process comments one by one
    comments_normalized = []
    for comment in comments_list:
        if isinstance(comment, str) and comment.strip():
            normalized = spacy_clean_normalize_single(comment, pipeline_dict_int, fallback_pipeline_int)
            if normalized:
                comments_normalized.append(normalized)

    comments = " ".join(comments_normalized)

    logger.info(" Completed LOV row %d/%d.", idx, total)
    return {"tags": tags, "comments": comments}


def preprocess_lov_data(input_frame: pd.DataFrame, pipeline_dict_int=None, fallback_pipeline_int=None) -> pd.DataFrame:
    total = len(input_frame)
    processed_rows: list[dict[str, Any]] = []

    for i, (_, row) in enumerate(input_frame.iterrows(), start=1):
        processed_rows.append(process_lov_data_row(row, i, total, pipeline_dict_int, fallback_pipeline_int))

    out_df = pd.DataFrame({
        "id": input_frame["id"] if "id" in input_frame.columns else list(range(total)),
        "tags": [r["tags"] for r in processed_rows],
        "comments": [r["comments"] for r in processed_rows],
    })
    logger.info("LOV processing complete: %d/%d.", len(out_df), total)
    return out_df


def process_all_from_input(
        input_data: Any,
        use_ner: bool = True,
        enable_filter: bool = True
) -> dict[str, list[Any]]:
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
        use_ner=use_ner, enable_filter=enable_filter
    )
    void_df = preprocess_void(
        df, pipeline_dict, fallback_pipeline
    )

    return {
        "id": remove_duplicates(combined_df["id"].tolist()),
        "title": remove_duplicates(combined_df["title"].tolist()),
        "lab": remove_duplicates(combined_df["lab"].tolist()),
        "lcn": remove_duplicates(sum(combined_df["lcn"].tolist(), [])),
        "lpn": remove_duplicates(sum(combined_df["lpn"].tolist(), [])),
        "curi": remove_duplicates(sum(combined_df["curi"].tolist(), [])),
        "puri": remove_duplicates(sum(combined_df["puri"].tolist(), [])),
        "voc": remove_duplicates(sum(combined_df["voc"].tolist(), [])),
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


def main(use_ner: bool = True, use_gpu: bool = False, enable_filter: bool = True) -> None:
    logger.info("Starting preprocessing workflow. NER enabled: %s, GPU enabled: %s, filter enabled: %s", use_ner,
                use_gpu, enable_filter)

    df = merge_dataset()
    logger.info("Merged dataset contains %d rows.", len(df))

    combined_df = preprocess_combined(df, pipeline_dict, fallback_pipeline, use_ner=use_ner,
                                      enable_filter=enable_filter)
    logger.info("After combined preprocessing: %d rows.", len(combined_df))

    void_df = preprocess_void(merge_void_dataset(), pipeline_dict, fallback_pipeline)
    logger.info("After void preprocessing: %d rows.", len(void_df))

    lov_raw = pd.read_json(os.path.join(RAW_DIR, "lov_cloud", "voc_cmt.json"))
    logger.info("Loaded LOV raw data: %d rows.", len(lov_raw))

    lov_data = preprocess_lov_data(lov_raw, pipeline_dict, fallback_pipeline)
    logger.info("After LOV preprocessing: %d rows.", len(lov_data))

    final_df = combine_with_void_and_lov_data(combined_df, void_df, lov_data)
    final_df = remove_empty_list_values(final_df)
    logger.info("Final merged DataFrame: %d rows.", len(final_df))

    output_path = os.path.join(PROCESSED_DIR, "combined.json")
    final_df.to_json(output_path, orient="records", lines=False)
    logger.info("Preprocessing complete. Saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset with optional NER, GPU, and filter.")
    parser.add_argument("--no-ner", action="store_true", help="Disable NER and set ner field to [].")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for spaCy pipelines if available.")
    parser.add_argument("--no-filter", action="store_true", help="Disable the filter checks for is_*_allowed.")
    args = parser.parse_args()

    main(
        use_ner=not args.no_ner,
        use_gpu=args.gpu,
        enable_filter=not args.no_filter
    )
