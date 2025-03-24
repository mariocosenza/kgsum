import logging
import os
import time

import ollama
import pandas as pd
import requests
from google import genai

# Define allowed categories
LOD_CATEGORY_NO_MULTIPLE_DOMAIN = {
    'geography', 'government', 'life_sciences', 'cross_domain',
    'linguistics', 'media', 'publications', 'social_networking'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_generate_content(g_client, description, max_retries=5, initial_wait=60):
    retries = 0
    wait_time = initial_wait
    while retries < max_retries:
        try:
            result = g_client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=(
                    f"Given the following description, find a category from this list. "
                    f"Only respond with the category and no other words. "
                    f"Be precise and use your reasoning. "
                    f"Use the same category format. "
                    f"Categories: {LOD_CATEGORY_NO_MULTIPLE_DOMAIN}. "
                    f"Description: {description}"
                )
            )
            # Normalize the result to a string.
            if isinstance(result, dict):
                # Check common keys for the response text.
                output = result.get("text", "").strip() or result.get("response", "").strip()
            elif isinstance(result, str):
                output = result.strip()
            else:
                output = str(result).strip()
            return output
        except Exception as e:
            logger.warning(
                f"Gemini server overloaded (attempt {retries + 1}/{max_retries}). Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)
            retries += 1
            wait_time *= 2  # Exponential backoff
    raise Exception("Max retries exceeded for generate_content (Gemini)")


def safe_generate_content_ollama(description):
    prompt = (
        f"Given the following description, find a category from this list. "
        f"Only respond with the category and no other words. "
        f"Be precise and use your reasoning. "
        f"Use the same category format. "
        f"Categories: {LOD_CATEGORY_NO_MULTIPLE_DOMAIN}. "
        f"Description: {description}"
    )

    try:
        response = ollama.generate(model="gemma3:12b", prompt=prompt)
        output = response['response'].strip()
        logger.info(f'Categories: {output}')
    except Exception as e:
        logger.error("Error calling Ollama model: %s", e)
        return ""

    return output


def get_zenodo_records(g_client, use_ollama=False):
    url = "https://zenodo.org/api/records"
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
    category_cache = {}

    calls_in_minute = 0
    minute_start = time.time()

    for hit in data.get("hits", {}).get("hits", []):
        metadata = hit.get("metadata", {})
        title = metadata.get("title", "")

        if "wikidata" in title.lower():
            logger.info(f"Skipping repository '{title}' because it contains 'wikidata' in the title.")
            continue

        description = metadata.get("description", "") or title
        record_link = hit.get("links", {}).get("self", "")

        logger.info(f"Processing repository: {title}")

        if title in category_cache:
            category = category_cache[title]
            logger.info(f"Using cached category for repository: {title}")
        else:
            if use_ollama:
                try:
                    result = safe_generate_content_ollama(description)
                    category = result.strip()
                except Exception as e:
                    logger.error(f"Failed to generate content for record '{title}' using Ollama: {e}")
                    category = "unknown"
            else:
                # Enforce rate limiting for Gemini: max 10 calls per minute
                if calls_in_minute >= 10:
                    elapsed = time.time() - minute_start
                    if elapsed < 60:
                        time.sleep(60 - elapsed)
                    minute_start = time.time()
                    calls_in_minute = 0

                try:
                    result = safe_generate_content(g_client, description)
                    category = result.strip()
                    calls_in_minute += 1
                except Exception as e:
                    logger.error(f"Failed to generate content for record '{title}' using Gemini: {e}")
                    category = "unknown"
            category_cache[title] = category

        records.append({
            "title": title,
            "description": description,
            "record_link": record_link,
            "category": category
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="record_link")
    df = df.sort_values(by="title", ascending=True)
    return df


def download_file_for_record(record_detail, download_folder):
    files = record_detail.get("files", [])
    min_size = 15 * 1024  # 15 KB
    max_size = 2 * 1024 * 1024 * 1024  # 2 GB
    candidate_file = None

    # Search for an .nt file first
    for file in files:
        file_name = file.get("key")
        if file_name and file_name.lower().endswith(".nt"):
            file_size = file.get("size", 0)
            if min_size <= file_size <= max_size:
                candidate_file = file
                break

    # If no qualifying .nt file, search for a .ttl file
    if candidate_file is None:
        for file in files:
            file_name = file.get("key")
            if file_name and file_name.lower().endswith(".ttl"):
                file_size = file.get("size", 0)
                if min_size <= file_size <= max_size:
                    candidate_file = file
                    break

    if candidate_file is None:
        logger.info("No qualifying file found for this record.")
        return None

    file_name = candidate_file.get("key")
    record_id = record_detail.get("id")
    if not record_id:
        record_link = record_detail.get("links", {}).get("self", "")
        if record_link:
            record_id = record_link.rstrip("/").split("/")[-1]
        else:
            logger.error("Record ID not found.")
            return None

    download_url = f"https://zenodo.org/record/{record_id}/files/{file_name}?download=1"
    logger.info(f"Downloading file {file_name} from {download_url}")

    response = requests.get(download_url, stream=True)
    response.raise_for_status()

    os.makedirs(download_folder, exist_ok=True)
    file_path = os.path.join(download_folder, file_name)

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return file_name


def process_zenodo_records_with_download(g_client, download_folder, output_csv_path, use_ollama=False):
    df = get_zenodo_records(g_client, use_ollama=use_ollama)
    file_names = []

    for index, row in df.iterrows():
        record_link = row["record_link"]
        try:
            detail_response = requests.get(record_link, timeout=600)
            detail_response.raise_for_status()
            record_detail = detail_response.json()
            file_name = download_file_for_record(record_detail, download_folder)
            if file_name is None:
                file_name = ""
        except Exception as e:
            logger.error(f"Failed to process record '{row['title']}': {e}")
            file_name = ""
        file_names.append(file_name)

    df["file_name"] = file_names
    df.to_csv(output_csv_path, index=False)
    return df


if __name__ == "__main__":
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    download_folder = "../data/downloads"
    output_csv_path = "../data/raw/zenodo_with_files.csv"
    use_ollama = True
    df_final = process_zenodo_records_with_download(client, download_folder, output_csv_path, use_ollama=use_ollama)
