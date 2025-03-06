import logging
import os
import time
import requests
import pandas as pd
from urllib.parse import quote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def save_to_csv(results, output_file="../data/raw/github_unique_repos_with_ttl_nt.csv"):
    if not results:
        logger.info("No results to save.")
        return
    df = pd.DataFrame(results)
    df.sort_values("repository", inplace=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(results)} unique repositories (sorted by title) to {output_file}")


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN is not set in the environment variables.")
        return

    file_extensions = ["ttl", "nt", "nq"]
    logger.info(f"Searching for files with extensions: {', '.join(file_extensions)}")

    results = search_github_files(file_extensions, token, max_pages=10)
    save_to_csv(results)


if __name__ == "__main__":
    main()
