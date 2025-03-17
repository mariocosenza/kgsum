import logging
import os
import time
from xml.dom import minidom

import pandas as pd
from google import genai

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


def predict_category_from_lod_description(limit=500) -> pd.DataFrame:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    df = pd.read_json('../../data/raw/lod-data.json')
    hit = 0
    miss = 0

    # Rate limiting: track calls and the start of the minute
    calls_in_minute = 0
    minute_start = time.time()
    records = []

    logger.info('Started processing lod')

    for col in df.columns:
        if hit + miss > limit:
            break
        logger.info(f'Processing id: {col}')
        # Enforce rate limiting: max 10 calls per minute
        if calls_in_minute >= 10:
            elapsed = time.time() - minute_start
            if elapsed < 60:
                time.sleep(60 - elapsed)
            minute_start = time.time()
            calls_in_minute = 0

        if df[col]['domain'] != '' and df[col]['domain'] not in 'cross_domain' and df[col][
            'domain'] not in 'user_generated':
            try:
                result = client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21",
                    contents=(
                        f"Given the following description and keywords, find a category given this data. "
                        f"Only respond with the category and no other words. "
                        f"Be precise and use your reasoning. "
                        f"Use the same category format. "
                        f"Categories: {LOD_CATEGORY_NO_MULTIPLE_DOMAIN}. "
                        f"Description: {df[col]['description']}"
                        f"Keywords: {df[col]['keywords']}. "
                    )
                )
            except Exception as e:
                result = ''
                calls_in_minute += 2
                logger.warning(e)

            result_dict = {
                'lod_category': df[col]['domain'],
                'predicted_category': result.text.strip(),
                'id': col
            }

            logger.info(f'Processed: {result_dict}')

            records.append(result_dict)

            if result.text.strip() in df[col]['domain']:
                hit += 1
            elif df[col]['domain'] not in '':
                miss += 1

            if miss >= 1:
                logger.info(f'Hit: {hit}, Miss: {miss}, Rate: {hit * 100 / miss}%')

            calls_in_minute += 1
    df = pd.DataFrame(records)

    logger.info(f'Category hit: {hit}')
    logger.info(f'Category miss: {miss}')
    return df


def predict_category_from_lod_svg(limit=500):
    doc = minidom.parse('../../data/raw/lod-cloud.svg')

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    df = pd.read_json('../../data/raw/lod-data.json')
    hit = 0
    miss = 0

    # Rate limiting: track calls and the start of the minute
    calls_in_minute = 0
    minute_start = time.time()
    records = []

    logger.info('Started processing lod')

    for elem in doc.getElementsByTagName('g'):
        try:
            int(elem.getAttribute('id'))
        except Exception as e:
            logger.warning('Skipping non it id')
            continue

        first_circle = elem.getElementsByTagName('circle')[0]
        col = str(first_circle.getElementsByTagName('title')[0].firstChild.nodeValue)

        if hit + miss > limit:
            break

        logger.info(f'Processing id: {col}')
        # Enforce rate limiting: max 10 calls per minute
        if calls_in_minute >= 10:
            elapsed = time.time() - minute_start
            if elapsed < 60:
                time.sleep(60 - elapsed)
            minute_start = time.time()
            calls_in_minute = 0

        if df[col]['domain'] not in 'cross_domain' and df[col][
            'domain'] not in 'user_generated':
            try:
                result = client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21",
                    contents=(
                        f"Given the following description and keywords, find a category given this data. "
                        f"Only respond with the category and no other words. "
                        f"Be precise and use your reasoning. "
                        f"Use the same category format. "
                        f"Categories: {LOD_CATEGORY_NO_MULTIPLE_DOMAIN}. "
                        f"Description: {df[col]['description']}"
                        f"Keywords: {df[col]['keywords']}. "
                    )
                )
            except Exception as e:
                result = ''
                calls_in_minute += 2
                logger.warning(e)

            lod_category = CATEGORIES_COLOR.get(first_circle.getAttribute('fill').lower())
            result_dict = {
                'lod_category': lod_category,
                'predicted_category': result.text.strip(),
                'id': col
            }

            logger.info(f'Processed: {result_dict}')

            records.append(result_dict)

            if result.text.strip() in lod_category:
                hit += 1
            else:
                miss += 1

            if miss >= 1:
                logger.info(f'Hit: {hit}, Miss: {miss}, Rate: {hit * 100 / miss}%')

            calls_in_minute += 1
    df = pd.DataFrame(records)

    logger.info(f'Category hit: {hit}')
    logger.info(f'Category miss: {miss}')
    return df


if __name__ == '__main__':
    predict_category_from_lod_svg().to_csv('../../data/raw/lod-gemini-svg.csv')
    # predict_category_from_lod_description().to_csv('../../data/raw/lod-gemini.csv')
