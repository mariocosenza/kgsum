import asyncio
import os
import aiohttp
import urllib.parse
import pandas as pd

from src.dataset_preparation import process_file_full_inplace, logger
from src.dataset_preparation_remote import process_endpoint_full_inplace
from src.lov_data_preparation import IS_URI
from src.predict_category import CategoryPredictor
from src.preprocessing import process_all_from_input

LOCAL_ENDPOINT = os.environ['LOCAL_ENDPOINT']

PREDICTOR : CategoryPredictor = CategoryPredictor.get_predictor()

async def _fetch_query(query, timeout=300):
    async with aiohttp.ClientSession() as session:
        async with session.post(LOCAL_ENDPOINT + '/statements', data={'update': query}, timeout=timeout) as response:
            return await response.text()

async def generate_profile(endpoint: None | str = None, file: None | str = None) -> dict:
    if file is not None:
        processed_data = process_all_from_input(process_file_full_inplace(file))
    elif endpoint is not None:
        processed_data = process_all_from_input(await process_endpoint_full_inplace(endpoint))
    else:
        return {
            'error': 'Upload a file or input a valid SPARQL endpoint'
        }

    return {
            'profile': create_profile(processed_data),
            'category': PREDICTOR.predict_category(processed_data)
    }

async def generate_and_store_profile(endpoint=None, file=None):
    row = await generate_profile(endpoint=endpoint, file=file)
    await store_profile(profile=row['profile'], category=row['category'])
    return row

async def generate_profile_from_store():
    dataset = pd.read_json('../data/processed/combined.json')
    for index, col in dataset.iterrows():
        print(col['id'])
        await store_profile(profile=create_profile(data=col), category=str(col['category']))

def create_profile(data: dict | pd.DataFrame | pd.Series) -> dict:
    if isinstance(data, pd.DataFrame):
        data = data.to_dict('records')
    return data

def _to_list(val):
    """Ensure property value is returned as a list."""
    if val is None:
        return []
    return val if isinstance(val, list) else [val]

async def store_profile(profile: dict, category: str):
    raw_id = profile.get('id')
    if not raw_id:
        logger.warning("Missing profile id. Skipping insertion.")
        return

    # If the raw_id is not a valid IRI, generate one from it.
    if not IS_URI.match(raw_id):
        base_iri = "http://example.org/resource/"
        encoded_id = urllib.parse.quote(raw_id, safe="")
        iri = base_iri + encoded_id
        logger.info(f"Generated IRI {iri} from raw id {raw_id}")
    else:
        iri = raw_id

    # Wrap the final IRI in angle brackets for SPARQL syntax.
    iri_formatted = f"<{iri}>"



    # Build main triples with proper literal quoting.
    triples = [f"{iri_formatted} rdf:type dcat:dataset"]

    for title in _to_list(profile.get('title')):
        triples.append(f'{iri_formatted} dcterms:title "{title}"')
    for language in _to_list(profile.get('language')):
        triples.append(f'{iri_formatted} dcterms:language "{language}"')
    for dsc in _to_list(profile.get('dsc')):
        triples.append(f'{iri_formatted} dcterms:description "{dsc}"')
    for creator in _to_list(profile.get('creator')):
        triples.append(f'{iri_formatted} dcterms:creator "{creator}"')
    for lic in _to_list(profile.get('license')):
        triples.append(f'{iri_formatted} dcterms:license "{lic}"')
    for sparql in _to_list(profile.get('sparql')):
        if IS_URI.match(sparql):
            triples.append(f'{iri_formatted} dcterms:endpointURL <{sparql}>')
    triples.append(f'{iri_formatted} dcterms:identifier "{raw_id}"')

    # Add category as a theme.
    triples.append(f'{iri_formatted} dcat:theme "{category}"')

    insert_data = " .\n".join(triples) + " ."

    query_main = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        INSERT DATA {{
        {insert_data}
        }}
        """.strip()

    try:
        await _fetch_query(query_main)
    except Exception as error:
        logger.warning(f"Cannot insert data with iri: {iri}. Error: {error}")
        return

    try:
        vocab_triples = ""
        for voc in _to_list(profile.get('voc')):
            if IS_URI.match(voc):
                vocab_triples += f"{iri_formatted} void:vocabulary <{voc}> .\n"

        keyword_triples = ""
        for tag in _to_list(profile.get('tags')):
            keyword_triples += f'{iri_formatted} dcat:keyword "{tag}" .\n'

        if vocab_triples or keyword_triples:
            query_vocab = f"""
                PREFIX void: <http://rdfs.org/ns/void#>
                PREFIX dcat: <http://www.w3.org/ns/dcat#>
                INSERT DATA {{
                {vocab_triples}
                {keyword_triples}
            }}
            """.strip()
            await _fetch_query(query_vocab)

        # Build subject triples.
        subject_triples = ""
        for subj in _to_list(profile.get('sbj')):
            if IS_URI.match(subj):
                subject_triples += f'{iri_formatted} dcterms:subject <{subj}> .\n'
        if subject_triples:
            query_subject = f"""
            PREFIX dcterms: <http://purl.org/dc/terms/>
            INSERT DATA {{
            {subject_triples}
            }}
            """.strip()
            await _fetch_query(query_subject)

    except Exception as error:
        logger.warning(f"Cannot insert vocabulary or subject data. Error: {error}")


if __name__ == '__main__':
    asyncio.run(generate_profile_from_store())