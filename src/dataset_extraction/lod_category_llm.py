import logging
import os
import time
from itertools import count

import pandas as pd
from google import genai

from src.util import LOD_CATEGORY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_category_from_lod_description() -> pd.DataFrame:
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
        logger.info(f'Processing id: {col}')
        # Enforce rate limiting: max 15 calls per minute
        if calls_in_minute >= 10:
            elapsed = time.time() - minute_start
            if elapsed < 60:
                time.sleep(60 - elapsed)
            minute_start = time.time()
            calls_in_minute = 0

        try:
            result = client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=(
                    f"Given the following description and keywords, find a category given this data. "
                    f"Only respond with the category and no other words. "
                    f"Be precise and use your reasoning. "
                    f"Use the same category format. "
                    f"Data identified by {col}. "
                    f"Categories: {LOD_CATEGORY}. "
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


if __name__ == '__main__':
    predict_category_from_lod_description().to_json('../../data/raw/lod-gemini.json')