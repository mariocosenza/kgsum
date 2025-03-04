import os
import time
import requests
import pandas as pd
from google import genai
import logging

LOD_CATEGORY = {
    'cross_domain', 'geography', 'government', 'life_sciences',
    'linguistics', 'media', 'publications', 'social_networking', 'user_generated'
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_zenodo_records(g_client):
    url = "https://zenodo.org/api/records"
    params = {
        "q": "",
        "f": ["file_type:nt", "file_type:ttl"],
        "page": 1,
        "size": 820,
        "sort": "bestmatch"
    }

    response = requests.get(url, params=params)
    data = response.json()

    records = []
    # For rate limiting: track calls and the start of the minute
    calls_in_minute = 0
    minute_start = time.time()

    for hit in data["hits"]["hits"]:
        metadata = hit.get("metadata", {})
        title = metadata.get("title", "")
        description = metadata.get("description", "")
        if description == "":
            description = title

        record_link = hit.get("links", {}).get("self", "")

        logger.log(msg=f"Processing repository: {title}", level=logging.INFO)

        # Rate limiting: If we've reached 15 calls, wait until a minute has passed
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
                f"Given the following description find a category from this list only respond with the category no other word. "
                f"Analyze smartly the semantic to give a precise response with the category. Categories {LOD_CATEGORY}. "
                f"Description: {description}"
            )
        )
        calls_in_minute += 1

        records.append({
            "title": title,
            "description": description,
            "record_link": record_link,
            "category": result.text.replace("\n", "")
        })

    # Create a DataFrame and sort by title alphabetically
    df = pd.DataFrame(records)
    df = df.sort_values(by="title", ascending=True)
    return df

if __name__ == "__main__":
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    df_zenodo = get_zenodo_records(client)
    df_zenodo.to_csv(path_or_buf='../data/raw/zenodo.csv', index=False)
