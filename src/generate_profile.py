import os

import aiohttp

from src.dataset_preparation import process_file_full_inplace
from src.dataset_preparation_remote import process_endpoint_full_inplace

LOCAL_ENDPOINT = os.environ['LOCAL_ENDPOINT']


async def _fetch_query(query, timeout=300):
    async with aiohttp.ClientSession() as session:
        async with session.post(LOCAL_ENDPOINT, data={'query': query}, timeout=timeout) as response:
            return await response.text()


def generate_profile(processed_data=None, endpoint=None, file=None):
    if file is not None:
        processed_data = process_file_full_inplace(file)
    elif endpoint is not None:
        processed_data = process_endpoint_full_inplace(endpoint)

    if processed_data is not None:
        return {
            'row': processed_data,
            'profile': ''
        }
    else:
        return


async def store_profile(profile):
    await _fetch_query("""INSERT DATA { { } }""")


async def generate_and_store_profile(endpoint=None, file=None):
    row = generate_profile(endpoint=endpoint, file=file)
    await store_profile(profile=row['profile'])

    return row['profile']


def generate_profile_from_store():
    return
