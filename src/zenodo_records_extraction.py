import os

import requests
import pandas as pd
from google import genai

LOD_CATEGORY = {'cross_domain', 'geography', 'government', 'life_sciences', 'linguistics', 'media', 'publications', 'social_networking', 'user_generated'}

def get_zenodo_records():
    url = "https://zenodo.org/api/records"
    params = {
        "q": "",
        "f": ["file_type:nt", "file_type:ttl"],
        "page": 1,
        "size": 2,
        "sort": "bestmatch"
    }

    # Send the GET request
    response = requests.get(url, params=params)
    data = response.json()

    records = []
    # Process each record in the response
    for hit in data["hits"]["hits"]:
        metadata = hit.get("metadata", {})
        title = metadata.get("title", "")
        description = metadata.get("description", "")
        record_link = hit.get("links", {}).get("self", "")

        records.append({
            "title": title,
            "description": description,
            "record_link": record_link,
            "category": client.models.generate_content(
                        model="gemini-2.0-flash", contents=f"Given the following description find a category from this list only respond with the category no other word. Analyze smartly the semantic to give a precise response with the category. Categories {LOD_CATEGORY}. Description: {description}"
                        ).text.replace("\n", '')
        })

    # Create a DataFrame and sort by title alphabetically
    df = pd.DataFrame(records)
    df = df.sort_values(by="title", ascending=True)
    return df


if __name__ == "__main__":
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    df_zenodo = get_zenodo_records()
    print(df_zenodo)
