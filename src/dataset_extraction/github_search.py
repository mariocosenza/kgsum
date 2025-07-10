import hashlib
import logging
import os
import time
from urllib.parse import quote

import ollama
import pandas as pd
import requests
from google import genai

from src.util import LOD_CATEGORY_NO_USER_DOMAIN

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
                    f"""Given the following README file, find a category from the given list. 
                    Only respond with the category and no other words. 
                    Be precise and use your reasoning. 
                    Use the same format for the categories as instructed. 
                    Categories: {LOD_CATEGORY_NO_USER_DOMAIN}. 
                    Description: {description}"""
                )
            )
            return result.text.strip()
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
        f"""Given the following README file, find a category from the given list. 
        Only respond with the category and no other words. 
        Be precise and use your reasoning. "
        Use the same format for the categories as instructed. 
        Categories: {LOD_CATEGORY_NO_USER_DOMAIN}. "
        Description: {description}"""
    )

    try:
        response = ollama.generate(model="gemma3:12b", prompt=prompt)
        output = response['response'].strip()
        logger.info(f'Category: {output}')
    except Exception as e:
        logger.error(f"Error calling Ollama model: {e}")
        return ""

    return output


def get_with_rate_limit(url, headers):
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 403 and "rate limit" in response.text.lower():
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_time = max(reset_time - time.time(), 0) + 5  # 5-second buffer
            logger.info(f"Rate limit exceeded. Sleeping for {sleep_time:.0f} seconds.")
            time.sleep(sleep_time)
        else:
            return response


def search_by_size_range(extension, token, size_range, max_pages=10):
    size_min, size_max = size_range
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    results = []
    query = f"extension:{extension} size:{size_min}..{size_max}"
    encoded_query = quote(query)

    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/search/code?q={encoded_query}&per_page=100&page={page}"
        logger.info(f"Querying for {extension} files of size {size_min}-{size_max} bytes, page {page}: {url}")
        response = get_with_rate_limit(url, headers)

        if response.status_code != 200:
            logger.error(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        items = data.get("items", [])
        if not items:
            break

        results.extend(items)

        if len(items) < 100:
            # Likely reached the last page for this query.
            break

        time.sleep(1)
    return results


def search_github_files(file_extensions, token, max_pages=10):
    all_results = []
    seen_repos = set()

    # Define finer size ranges (in bytes) from 10KB to 3GB.
    size_ranges = [
        (10241, 20480),  # 10KB to 20KB
        (20481, 51200),  # 20KB to 50KB
        (51201, 102400),  # 50KB to 100KB
        (102401, 204800),  # 100KB to 200KB
        (204801, 512000),  # 200KB to 500KB
        (512001, 1048576),  # 500KB to 1MB
        (1048577, 2097152),  # 1MB to 2MB
        (2097153, 5242880),  # 2MB to 5MB
        (5242881, 10485760),  # 5MB to 10MB
        (10485761, 20971520),  # 10MB to 20MB
        (20971521, 52428800),  # 20MB to 50MB
        (52428801, 104857600),  # 50MB to 100MB
        (104857601, 209715200),  # 100MB to 200MB
        (209715201, 524288000),  # 200MB to 500MB
        (524288001, 1073741824),  # 500MB to 1GB
        (1073741825, 2147483648),  # 1GB to 2GB
        (2147483649, 3221225472)  # 2GB to 3GB
    ]

    for ext in file_extensions:
        logger.info(f"Searching for '{ext}' files across size ranges.")
        for size_range in size_ranges:
            items = search_by_size_range(ext, token, size_range, max_pages)
            for item in items:
                repo = item.get("repository", {})
                repo_name = repo.get("full_name")
                if not repo_name or repo_name in seen_repos:
                    continue
                seen_repos.add(repo_name)
                repo_info = {
                    "repository": repo_name,
                    "repo_url": repo.get("html_url"),
                    "file_url": item.get("html_url")
                }
                all_results.append(repo_info)

    logger.info(f"Found {len(all_results)} unique repositories for extensions: {', '.join(file_extensions)}")
    return all_results


def save_to_csv(results, output_file="../../data/raw/github_unique_repos_with_ttl_nt.csv"):
    if not results:
        logger.info("No results to save.")
        return None
    df = pd.DataFrame(results)
    df = df.sort_values("repository")
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(results)} unique repositories (sorted by title) to {output_file}")
    return df


def download_and_predict(g_client, download_folder, output_file="../data/raw/github_unique_repos_with_ttl_nt.csv",
                         use_ollama=False):
    id_int = 2000
    token = os.environ.get("GITHUB_TOKEN")
    df = pd.read_csv("../data/raw/github_unique_repos_with_ttl_nt.csv")

    # Ensure the 'category' column exists
    if 'category' not in df.columns:
        df['category'] = ""

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    os.makedirs(download_folder, exist_ok=True)

    for index, row in df.iterrows():
        repo = row['repository']
        try:
            resp = get_with_rate_limit(f'https://raw.githubusercontent.com/{repo}/main/README.md', headers)
            if resp.status_code == 200:
                if id_int == 3000 and not use_ollama:
                    res = input('Change your IP to continue with free API or confirm current key (True to change IP):')
                    if bool(res):
                        g_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

                # Predict the category
                if use_ollama:
                    predicted_category = safe_generate_content_ollama(description=resp.text[:100000]).strip()
                else:
                    predicted_category = safe_generate_content(g_client, description=resp.text[:100000]).strip()

                # Update the DataFrame with the new category value
                df.loc[index, 'category'] = predicted_category
                logger.info(f"Category predicted for {repo}: {predicted_category}")
                id_int += 1

                # Process file_url for downloading RDF dump
                file_name = str(row['file_url'])
                file_name = file_name.replace('https://github.com/', '').replace('/blob', '')
                rdf_dump_resp = get_with_rate_limit(f'https://raw.githubusercontent.com/{file_name}', headers)
                if rdf_dump_resp.status_code == 200:
                    category_folder = os.path.join(download_folder, predicted_category)
                    os.makedirs(category_folder, exist_ok=True)
                    file_path = os.path.join(category_folder,
                                             f'{id_int}-{hashlib.sha256(repo.encode()).hexdigest()}.rdf')
                    with open(file_path, "wb") as f:
                        for chunk in rdf_dump_resp.iter_content(chunk_size=8192):
                            f.write(chunk)
        except Exception as e:
            logger.error(f"An error occurred while processing repository {repo}. Check the final output! Error: {e}")

    df = df.sort_values("repository")
    df.to_csv(output_file, index=False)


def main(use_gemini=True):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN is not set in the environment variables.")
        return
    download_folder = "../data/raw/rdf_dump"
    file_extensions = ["ttl", "nt", "nq"]
    logger.info(f"Searching for files with extensions: {', '.join(file_extensions)}")
    results = search_github_files(file_extensions, token, max_pages=10)
    save_to_csv(results)
    download_and_predict(client, download_folder, use_ollama=not use_gemini)





if __name__ == "__main__":
    main()

