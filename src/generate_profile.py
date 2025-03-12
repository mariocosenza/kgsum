import asyncio
import os
import aiohttp
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from src.dataset_preparation import process_file_full_inplace
from src.dataset_preparation_remote import process_endpoint_full_inplace
from src.vocabulary_extraction import IS_URI
from src.predict_category import predict_category_multi

LOCAL_ENDPOINT = os.environ['LOCAL_ENDPOINT']


async def _fetch_query(query, timeout=300):
    async with aiohttp.ClientSession() as session:
        async with session.post(LOCAL_ENDPOINT, data={'query': query}, timeout=timeout) as response:
            return await response.text()


def generate_profile(endpoint=None, file=None) -> dict:
    if file is not None:
        processed_data = process_file_full_inplace(file)
    elif endpoint is not None:
        processed_data = process_endpoint_full_inplace(endpoint)
    else:
        return {
            'error': 'Upload a file or input a valid SPARQL endpoint'
        }

    return {
            'profile': create_profile(processed_data),
            'category':  predict_category_multi(processed_data)
    }



async def store_profile(profile: pd.DataFrame, category: str):
    iri = profile['id']
    try:
        await _fetch_query(f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        INSERT DATA {{
        {iri} rdf:type dcat:dataset .
        {iri} dcterms:title "{profile['title']}" .
        {iri} dcterms:language "{'language'}" .
        {iri} dcterms:description "{profile['dsc']}" .
        {iri} dcterms:creator "{profile['creator']}" .
        {iri} dcterms:license "{profile['license']}" .
        {iri} dcterms:endpointURL "{profile['sparql']}" .
        {iri} dcat:theme "{category}" .
        }}
        """)
    except:
        return

    triple = ''
    keywords = ''
    subjects = ''
    for voc in profile['voc']:
        if IS_URI.match(voc):
            triple = triple + f"{iri} void:vocabulary <{voc}> . \n"

    for keyword in profile['tags']:
        keywords = keywords + f'{iri} dcat:keyword "{keyword}" .\n '

    for subject in profile['sbj']:
        if IS_URI.match(subject):
            subjects = subjects + f'{iri} dcterms:subject <{subject}> .\n '

    try:
        await _fetch_query(f"""
                 PREFIX void: <http://rdfs.org/ns/void#>
                 PREFIX dcat: <http://www.w3.org/ns/dcat#>
                 INSERT DATA {{
                     {triple}
                     {keywords}
                 }}
                """)
    except:
        pass

    try:
        await _fetch_query(f"""
                 PREFIX dcterms: <http://purl.org/dc/terms/>
                 INSERT DATA {{
                     {subjects}
                 }}
                 """)
    except:
        pass

async def generate_and_store_profile(endpoint=None, file=None):
    row = generate_profile(endpoint=endpoint, file=file)
    await store_profile(profile=row['profile'], category=row['category'])
    return row['profile']

async def generate_profile_from_store():
    dataset = pd.read_json('../data/processed/combined.json')
    for row in dataset.iterrows():
        await store_profile(profile=create_profile(data=row), category=row['category'])

def create_profile(data: dict | tuple) -> pd.DataFrame:
    return DataFrame()

if __name__ == '__main__':
    asyncio.run(generate_profile_from_store())