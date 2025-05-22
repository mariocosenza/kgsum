import hashlib
import logging
import os
import re
import time

import ollama
import pandas as pd
import requests
from google import genai

from src.util import LOD_CATEGORY_NO_USER_DOMAIN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limit configuration for Zenodo API calls.
ZENODO_RATE_LIMIT = 1.0  # calls per second
_last_zenodo_call = 0


def sanitize_folder_name(name: str) -> str:
    # Remove illegal characters: < > : " / \ | ? *
    name = re.sub(r'[<>:"/\\|?*]', '', name).strip()
    # If name is excessively long, trim it (e.g., to 50 characters)
    if len(name) > 50:
        name = name[:50].strip()
    # If the resulting name is empty, return a default value.
    return name or "unknown"


def extract_category(text: str) -> str:
    # Look for text='some_category' or text="some_category"
    match = re.search(r"text=['\"]([^'\"]+)['\"]", text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    # Fallback: if the output starts with "candidates=" or is excessively long, use "unknown"
    if text.startswith("candidates=") or len(text) > 50:
        return "unknown"
    return text.strip()


def zenodo_get(*args, **kwargs):
    global _last_zenodo_call, ZENODO_RATE_LIMIT
    max_retries = 5
    response = None
    for attempt in range(max_retries):
        now = time.time()
        wait = max(0.0, (1.0 / ZENODO_RATE_LIMIT) - (now - _last_zenodo_call))
        if wait:
            time.sleep(wait)
        response = requests.get(*args, **kwargs)
        _last_zenodo_call = time.time()

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            try:
                wait_time = float(retry_after) if retry_after is not None else 60.0
            except ValueError:
                wait_time = 60.0
            logger.warning(
                f"Received 429 Too Many Requests. Retrying after {wait_time} seconds (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(wait_time)
        else:
            rate_limit_header = response.headers.get("X-RateLimit-Limit")
            if rate_limit_header is not None:
                try:
                    new_limit = float(rate_limit_header)
                    if new_limit > 0 and new_limit != ZENODO_RATE_LIMIT:
                        logger.info(
                            f"X-RateLimit-Limit header indicates a new rate limit: {new_limit} calls per second."
                        )
                        ZENODO_RATE_LIMIT = new_limit
                except Exception as ex:
                    logger.warning(f"Unable to parse X-RateLimit-Limit header: {ex}")
            return response

    if response is not None:
        response.raise_for_status()
    else:
        raise Exception("Zenodo API call failed after maximum retries.")


def safe_generate_content(g_client, description, max_retries=5, initial_wait=60):
    retries = 0
    wait_time = initial_wait
    while retries < max_retries:
        try:
            result = g_client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=(
                    f"""Given the following description, find a category from this list. 
                    Only respond with the category and no other words. 
                    Be precise and use your reasoning. 
                    Use the same category format. 
                    If the predicted category is publication, try if other categories are more appropriate and use one of them instead. 
                    Categories: {LOD_CATEGORY_NO_USER_DOMAIN}. 
                    Description: {description}"""
                )
            )
            if isinstance(result, dict):
                output = result.get("text", "").strip() or result.get("response", "").strip()
            elif isinstance(result, str):
                output = result.strip()
            else:
                output = str(result).strip()
            predicted = extract_category(output)
            logger.info(f"Predicted category (Gemini): {predicted}")
            return predicted
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
        f"""Given the following description, find a category from this list. "
        Only respond with the category and no other words. Do not add any other words for any reason. "
        Be precise and use your reasoning. "
        Use the same category format. "
        If the predicted category is publication, try if other categories are more appropriate and use one of them instead. "
        Categories: {LOD_CATEGORY_NO_USER_DOMAIN}. "
        Description: {description}"""
    )

    try:
        # Added a 2-minute (120 seconds) timeout parameter
        response = ollama.generate(model="gemma3:12b", prompt=prompt, timeout=120)
        output = response['response'].strip()
        logger.info(f"Raw output from Ollama: {output}")
        predicted = extract_category(output)
        logger.info(f"Predicted category (Ollama): {predicted}")
        return predicted
    except Exception as e:
        logger.error("Error calling Ollama model: %s", e)
        return ""


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

    response = zenodo_get(url, params=params, timeout=600)
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
            logger.info(f"Using cached category for repository: {title} - {category}")
        else:
            if use_ollama:
                try:
                    result = safe_generate_content_ollama(description)
                    category = result.strip()
                except Exception as e:
                    logger.error(f"Failed to generate content for record '{title}' using Ollama: {e}")
                    category = "unknown"
            else:
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
    # Updated size limits: minimum 10KB and maximum 3GB
    min_size = 10 * 1024  # 10 KB
    max_size = 3 * 1024 * 1024 * 1024  # 3 GB

    candidate_files = []
    for file in files:
        file_name = file.get("key")
        if file_name and (file_name.lower().endswith(".nt") or file_name.lower().endswith(".ttl")):
            file_size = file.get("size", 0)
            if min_size <= file_size <= max_size:
                candidate_files.append(file)

    if not candidate_files:
        logger.info("No qualifying file found for this record.")
        return None

    candidate_file = None
    for file in candidate_files:
        if file.get("key").lower().endswith(".nt"):
            candidate_file = file
            break
    if candidate_file is None:
        candidate_file = candidate_files[0]

    # If the selected candidate is named README, try to find an alternative.
    base_name, ext = os.path.splitext(candidate_file.get("key"))
    if base_name.lower() == "readme":
        logger.info("Candidate file is README. Searching for an alternative file...")
        alternative = None
        for file in candidate_files:
            bname, _ = os.path.splitext(file.get("key"))
            if bname.lower() != "readme":
                alternative = file
                break
        if alternative:
            candidate_file = alternative
        else:
            logger.info("No alternative file found. Using README file.")

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

    response = zenodo_get(download_url, stream=True)
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
    file_indices = []
    current_index = 1000  # starting index

    for idx, row in df.iterrows():
        record_link = row["record_link"]
        try:
            # Extract and sanitize the category to use for the folder name.
            category = sanitize_folder_name(str(row["category"]))
            record_folder = os.path.join(download_folder, category)

            # Retrieve record details.
            detail_response = zenodo_get(record_link, timeout=600)
            detail_response.raise_for_status()
            record_detail = detail_response.json()

            downloaded_file_name = download_file_for_record(record_detail, record_folder)
            if downloaded_file_name:
                # Create a new file name based on the progressive index.
                _, ext = os.path.splitext(downloaded_file_name)
                new_file_name = f"{current_index}-{hashlib.sha256(record_link.encode()).hexdigest()}.rdf"
                original_path = os.path.join(record_folder, downloaded_file_name)
                new_path = os.path.join(record_folder, new_file_name)
                os.rename(original_path, new_path)
                file_names.append(new_file_name)
                file_indices.append(current_index)
                current_index += 1
            else:
                file_names.append("")
                file_indices.append(None)
        except Exception as e:
            logger.error(f"Failed to process record '{row['title']}': {e}")
            file_names.append("")
            file_indices.append(None)

    df["file_name"] = file_names
    df["file_index"] = file_indices
    df.to_csv(output_csv_path, index=False)
    return df


if __name__ == "__main__":
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    download_folder = "../../data/raw/rdf_dump"
    output_csv_path = "../../data/raw/zenodo_with_files.csv"
    use_ollama = False
    df_final = process_zenodo_records_with_download(client, download_folder, output_csv_path, use_ollama=use_ollama)
    # Alternatively:
    # get_zenodo_records(client, use_ollama=use_ollama).to_csv(output_csv_path, index=False)
