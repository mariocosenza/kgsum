import logging
import os
import time

import pandas as pd
import requests
from google import genai

LOD_CATEGORY = {
    'cross_domain', 'geography', 'government', 'life_sciences',
    'linguistics', 'media', 'publications', 'social_networking', 'user_generated'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_zenodo_records(g_client):
    url = "https://zenodo.org/api/records"
    # Embed the file_type filter directly into the q parameter
    params = [
        ("q", ""),
        ("page", 1),
        ("file_type", "ttl"),
        ("file_type", "nt"),
        ("size", 820),
        ("sort", "newest")
    ]

    response = requests.get(url, params=params, timeout=600)
    response.raise_for_status()
    data = response.json()

    records = []
    # Rate limiting: track calls and the start of the minute
    calls_in_minute = 0
    minute_start = time.time()

    for hit in data.get("hits", {}).get("hits", []):
        metadata = hit.get("metadata", {})
        title = metadata.get("title", "")
        description = metadata.get("description", "") or title
        record_link = hit.get("links", {}).get("self", "")

        logger.info(f"Processing repository: {title}")

        # Enforce rate limiting: max 15 calls per minute
        if calls_in_minute >= 15:
            elapsed = time.time() - minute_start
            if elapsed < 60:
                time.sleep(60 - elapsed)
            minute_start = time.time()
            calls_in_minute = 0

        # Generate the category using Gemini
        result = g_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                f"Given the following description, find a category from this list. "
                f"Only respond with the category and no other words. "
                f"Be precise and use your reasoning. "
                f"Use the same category format. "
                f"Categories: {LOD_CATEGORY}. "
                f"Description: {description}"
            )
        )
        calls_in_minute += 1

        records.append({
            "title": title,
            "description": description,
            "record_link": record_link,
            "category": result.text.strip()
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="title", ascending=True)
    return df

if __name__ == "__main__":
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    df_zenodo = get_zenodo_records(client)
    df_zenodo.to_csv(path_or_buf='../data/raw/zenodo.csv', index=False)
