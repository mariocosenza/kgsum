import logging
import os
import time
from xml.dom import minidom

import ollama
import pandas as pd
from google import genai

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from src.util import LOD_CATEGORY_NO_MULTIPLE_DOMAIN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORIES_COLOR: dict[str, str] = {
    '#c8a788': 'cross_domain',
    '#29c9cc': 'geography',
    '#f6b33c': 'government',
    '#db777f': 'life_sciences',
    '#36bc8d': 'linguistics',
    '#6372c7': 'media',
    '#bcb582': 'publications',
    '#b5b5b5': 'social_networking',
    '#d84d8c': 'user_generated'
}

def build_prompt(description, keywords):
    return (
        f"""Given the following description and keywords, find a category for the given data. 
        Only respond with the category and no other words. 
        Be precise and use your reasoning. 
        Use the same category format. 
        Categories: {LOD_CATEGORY_NO_MULTIPLE_DOMAIN}. 
        Description: {description}
        Keywords: {keywords}. """
    )

def call_gemini_with_retries(client, description, keywords, col, max_calls, calls_in_minute, minute_start):
    # Rate limiting (10 per min)
    if calls_in_minute[0] >= max_calls:
        elapsed = time.time() - minute_start[0]
        if elapsed < 60:
            time.sleep(60 - elapsed)
        minute_start[0] = time.time()
        calls_in_minute[0] = 0

    max_retries = 3
    retry_count = 0
    retry_wait = 60
    prompt = build_prompt(description, keywords)
    result = None

    while retry_count <= max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=prompt
            )
            result = response.text.strip().lower()
            break
        except Exception as e:
            error_str = str(e)
            logger.warning(f"Error with ID {col}: {error_str}")
            calls_in_minute[0] += 1

            if "429" in error_str:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.info(
                        f"Rate limit hit. Waiting {retry_wait} seconds before retry {retry_count}/{max_retries}")
                    time.sleep(retry_wait)
                    retry_wait *= 2
                else:
                    logger.warning(f"Max retries reached for ID {col}. Skipping.")
            else:
                logger.warning(f"Non-rate limit error. Skipping ID {col}.")
                break

    # Always increment calls if any attempt made
    calls_in_minute[0] += 1
    return result

def safe_generate_content_ollama(description, keywords) -> str:
    prompt = build_prompt(description, keywords)
    try:
        response = ollama.generate(model="gemma3:12b", prompt=prompt)
        output = response['response'].strip().lower()
        logger.info(f'Categories: {output}')
    except Exception as e:
        logger.error(f"Error calling Ollama model: {e}")
        return ""
    return output

def _category_prediction_core(records_iter, df, client, use_ollama, lod_category_func=None, limit=500):
    hit, miss = 0, 0
    calls_in_minute = [0]
    minute_start = [time.time()]
    records = []

    for col, meta in records_iter:
        if hit + miss > limit:
            break
        logger.info(f'Processing id: {col}')

        domain = df[col]['domain']
        if domain not in ('', 'cross_domain', 'user_generated'):
            if not use_ollama:
                result = call_gemini_with_retries(
                    client, df[col]['description'], df[col]['keywords'],
                    col, 10, calls_in_minute, minute_start
                )
                if result is None:
                    logger.warning(f"Skipping item {col} due to API errors")
                    continue
            else:
                result = safe_generate_content_ollama(
                    description=df[col]['description'],
                    keywords=df[col]['keywords']
                ).strip()

            lod_category = domain if lod_category_func is None else lod_category_func(meta)
            result_dict = {
                'lod_category': domain,
                'predicted_category': result,
                'id': col
            }

            logger.info(f'Processed: {result_dict}')
            records.append(result_dict)

            if result in lod_category:
                hit += 1
            else:
                miss += 1

            if miss >= 1:
                logger.info(
                    f'Hit: {hit}, Miss: {miss}, Rate: {hit * 100 / (hit + miss)}%')

    return pd.DataFrame(records), hit, miss

def predict_category_from_lod_description(limit=500, use_ollama=False) -> pd.DataFrame:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    df = pd.read_json('../../data/raw/lod-data.json')
    records_iter = ((col, None) for col in df.columns)
    result_df, hit, miss = _category_prediction_core(records_iter, df, client, use_ollama, None, limit)
    logger.info(f'Category hit: {hit}')
    logger.info(f'Category miss: {miss}')
    return result_df

def predict_category_from_lod_svg(limit=500, use_ollama=False) -> pd.DataFrame:
    doc = minidom.parse('../../data/raw/lod-cloud.svg')
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    df = pd.read_json('../../data/raw/lod-data.json')
    # Build an iterator of (col, meta) with the svg element and fill color
    records_iter = []
    for elem in doc.getElementsByTagName('g'):
        try:
            int(elem.getAttribute('id'))
        except Exception:
            logger.warning('Skipping non int id')
            continue
        first_circle = elem.getElementsByTagName('circle')[0]
        col = str(first_circle.getElementsByTagName('title')[0].firstChild.nodeValue)
        for id_col in df.columns:
            if col in df[id_col]['title']:
                col = id_col
                break
        meta = {'fill': first_circle.getAttribute('fill')}
        records_iter.append((col, meta))

    # Map fill color to canonical category
    def lod_category_func(meta):
        return CATEGORIES_COLOR.get(meta['fill'].lower(), "")

    result_df, hit, miss = _category_prediction_core(records_iter, df, client, use_ollama, lod_category_func, limit)
    logger.info(f'Category hit: {hit}')
    logger.info(f'Category miss: {miss}')
    return result_df

def calculate_metrics_sklearn(df: pd.DataFrame, average: str = "macro"):
    y_true = df['lod_category']
    y_pred = df['predicted_category']

    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }

if __name__ == '__main__':
    # df = predict_category_from_lod_svg().to_csv('../../data/raw/lod-gemini-svg.csv')
    df = predict_category_from_lod_description(use_ollama=False)
    df.to_csv('../../data/raw/lod-gemini.csv')

    metrics = calculate_metrics_sklearn(df)
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print("Classification report:\n", metrics["classification_report"])
