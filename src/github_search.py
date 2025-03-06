import logging

import requests
import csv
import time
import os
from urllib.parse import quote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_github_files(file_extensions, token, max_pages=10):
    results = []
    seen_repos = set()
    query = " OR ".join([f"extension:{ext}" for ext in file_extensions])

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/search/code?q={quote(query)}&per_page=100&page={page}"

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            logger.info(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()

        if "items" not in data:
            logger.info("No items found in response")
            break

        # Process each file
        for item in data["items"]:
            repo_name = item["repository"]["full_name"]

            # Skip if we've already seen this repository
            if repo_name in seen_repos:
                continue

            seen_repos.add(repo_name)

            file_info = {
                "repo_name": repo_name,
                "file_name": item["name"],
                "file_path": item["path"],
                "html_url": item["html_url"],
                "raw_url": item["html_url"].replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            }
            results.append(file_info)

        if len(data["items"]) < 100:
            break

        # Respect GitHub API rate limits
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        if remaining < 5:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            sleep_time = max(reset_time - time.time(), 0) + 1
            logger.info(f"Rate limit almost reached. Sleeping for {sleep_time:.0f} seconds.")
            time.sleep(sleep_time)
        else:
            time.sleep(1)

    logger.info(f"Found {len(results)} unique repositories with TTL/NT/NQ files")
    return results


def save_to_csv(results, output_file="../data/raw/github_unique_repos_with_ttl_nt.csv"):
    if not results:
        logger.info("No results to save")
        return

    fieldnames = results[0].keys()

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved {len(results)} unique repositories to {output_file}")


def main():
    # GitHub Personal Access Token (set as environment variable for security)
    token = os.environ.get("GITHUB_TOKEN")

    # File extensions to search for
    file_extensions = ["ttl", "nt", "nq"]

    logger.info(f"Searching for files with extensions: {', '.join(file_extensions)}")
    results = search_github_files(file_extensions, token)

    save_to_csv(results)


if __name__ == "__main__":
    main()