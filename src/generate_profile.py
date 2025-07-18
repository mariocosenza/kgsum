import asyncio
import os
import urllib.parse
from typing import Any

import aiohttp
import pandas as pd

from src.dataset_preparation import process_file_full_inplace, logger
from src.dataset_preparation_remote import process_endpoint_full_inplace
from src.lov_data_preparation import IS_URI
from src.predict_category import CategoryPredictor
from src.preprocessing import process_all_from_input

LOCAL_ENDPOINT = os.environ['LOCAL_ENDPOINT']
PREDICTOR: CategoryPredictor | None = None


def load_predictor():
    global PREDICTOR
    PREDICTOR = CategoryPredictor.get_predictor()


async def _update_query(query: str, timeout: int = 300) -> str:
    """Execute SPARQL update query against local endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    LOCAL_ENDPOINT + '/statements',
                    data={'update': query},
                    timeout=timeout
            ) as response:
                return await response.text()
    except asyncio.TimeoutError:
        logger.error(f"Query timeout after {timeout} seconds")
        raise
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise


async def generate_profile(endpoint: str | None = None, file: str | None = None) -> dict[str, Any]:
    """Generate profile from either file or endpoint."""
    try:
        if file is not None:
            processed_data = process_all_from_input(process_file_full_inplace(file))
        elif endpoint is not None:
            processed_data = process_all_from_input(await process_endpoint_full_inplace(endpoint))
        else:
            return {
                'error': 'Upload a file or input a valid SPARQL endpoint'
            }

        profile = create_profile(processed_data)

        # Ensure predictor is loaded
        if PREDICTOR is None:
            load_predictor()

        profile['category'] = PREDICTOR.predict_category(processed_data)
        return profile

    except Exception as e:
        logger.error(f"Profile generation failed: {e}")
        return {
            'error': f'Profile generation failed: {str(e)}'
        }


async def generate_and_store_profile(
        endpoint: str | None = None,
        file: str | None = None,
        base_uri: str = "http://localhost:8000/"
) -> dict[str, Any]:
    """Generate profile and store it in the triplestore."""
    try:
        row = await generate_profile(endpoint=endpoint, file=file)

        if 'error' in row:
            return row

        await store_profile(profile=row, category=row['category'], base_iri=base_uri)
        return row

    except Exception as e:
        logger.error(f"Profile generation and storage failed: {e}")
        return {
            'error': f'Profile generation and storage failed: {str(e)}'
        }


async def generate_profile_from_store(base_url: str = "https://exemple.org"):
    """Generate profiles from stored dataset."""
    try:
        dataset = pd.read_json('../data/processed/combined.json')
        for index, col in dataset.iterrows():
            try:
                print(f"Processing: {col['id']}")
                profile = create_profile(data=col)
                await store_profile(
                    profile=profile,
                    category=str(col['category']),
                    base_iri=base_url
                )
            except Exception as e:
                logger.warning(f"Failed to process row {index}: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to process dataset: {e}")
        raise


def create_profile(data: dict[str, Any] | pd.DataFrame | pd.Series) -> dict[str, Any]:
    """Create profile from input data."""
    try:
        if isinstance(data, pd.DataFrame):
            data = data.dropna()
            data = data.drop_duplicates()
            data = data.to_dict('records')
        elif isinstance(data, pd.Series):
            data = data.dropna().to_dict()

        return data if isinstance(data, dict) else {}

    except Exception as e:
        logger.error(f"Profile creation failed: {e}")
        return {}


def _flatten_and_stringify(val: Any) -> list[str]:
    """
    Flatten nested lists and convert all items to strings.
    This fixes the 'expected string or bytes-like object, got list' error.
    """
    if val is None:
        return []

    def _flatten_recursive(item: Any) -> list[str]:
        if isinstance(item, list):
            result = []
            for subitem in item:
                result.extend(_flatten_recursive(subitem))
            return result
        else:
            # Convert to string and filter out empty values
            item_str = str(item).strip() if item is not None else ""
            return [item_str] if item_str else []

    if isinstance(val, list):
        return _flatten_recursive(val)
    else:
        val_str = str(val).strip() if val is not None else ""
        return [val_str] if val_str else []


def _extract_first_valid_uri(val: Any) -> str:
    """
    Extract the first valid URI from a value that might be a list, string, or other type.
    Returns empty string if no valid URI is found.
    """
    if val is None:
        return ""

    # If it's a list, try to find the first valid URI
    if isinstance(val, list):
        for item in val:
            if item is not None:
                item_str = str(item).strip()
                if item_str and IS_URI.match(item_str):
                    return item_str
        # If no valid URI found in list, return the first non-empty item as string
        for item in val:
            if item is not None:
                item_str = str(item).strip()
                if item_str:
                    return item_str
        return ""
    else:
        # Single value - convert to string
        val_str = str(val).strip()
        return val_str if val_str else ""


def _escape_sparql_literal(value: str) -> str:
    """Escape special characters in SPARQL literals."""
    if not isinstance(value, str):
        value = str(value)

    # Escape quotes and other special characters
    value = value.replace('\\', '\\\\')  # Escape backslashes first
    value = value.replace('"', '\\"')  # Escape double quotes
    value = value.replace('\n', '\\n')  # Escape newlines
    value = value.replace('\r', '\\r')  # Escape carriage returns
    value = value.replace('\t', '\\t')  # Escape tabs

    return value


async def store_profile(
        profile: dict[str, Any],
        category: str,
        base_iri: str = "http://example.org/resource/"
) -> None:
    """Store profile data in triplestore with proper error handling."""

    raw_id = profile.get('id')
    if not raw_id:
        logger.warning("Missing profile id. Skipping insertion.")
        return

    # Initialize iri to avoid "might be referenced before assignment" error
    iri = ""

    try:
        # Extract the actual ID value (handles both lists and single values)
        raw_id_str = _extract_first_valid_uri(raw_id)

        if not raw_id_str:
            logger.warning(f"No valid ID found in raw_id: {raw_id}. Skipping insertion.")
            return

        logger.info(f"Extracted ID: '{raw_id_str}' from raw_id: {raw_id}")

        # Generate IRI - preserve original form if it's already a valid URI
        if IS_URI.match(raw_id_str):
            # Keep the original URI as-is
            iri = raw_id_str
            logger.info(f"Using original URI as IRI: {iri}")
        else:
            # Generate IRI from raw_id
            encoded_id = urllib.parse.quote(raw_id_str, safe="")
            iri = base_iri + encoded_id
            logger.info(f"Generated IRI {iri} from raw id {raw_id_str}")

        # Wrap the final IRI in angle brackets for SPARQL syntax
        iri_formatted = f"<{iri}>"

        # Build main triples with proper literal escaping
        triples = [f"{iri_formatted} rdf:type dcat:dataset"]

        # Process basic metadata fields
        for title in _flatten_and_stringify(profile.get('title')):
            if title:
                escaped_title = _escape_sparql_literal(title)
                triples.append(f'{iri_formatted} dcterms:title "{escaped_title}"')

        for language in _flatten_and_stringify(profile.get('language')):
            if language and language != 'UNKNOWN':
                escaped_lang = _escape_sparql_literal(language)
                triples.append(f'{iri_formatted} dcterms:language "{escaped_lang}"')

        for dsc in _flatten_and_stringify(profile.get('dsc')):
            if dsc:
                escaped_dsc = _escape_sparql_literal(dsc)
                triples.append(f'{iri_formatted} dcterms:description "{escaped_dsc}"')

        for creator in _flatten_and_stringify(profile.get('creator')):
            if creator:
                escaped_creator = _escape_sparql_literal(creator)
                triples.append(f'{iri_formatted} dcterms:creator "{escaped_creator}"')

        for lic in _flatten_and_stringify(profile.get('license')):
            if lic:
                escaped_lic = _escape_sparql_literal(lic)
                triples.append(f'{iri_formatted} dcterms:license "{escaped_lic}"')

        # Process URIs (sparql endpoints and connections) - preserve original form
        for sparql in _flatten_and_stringify(profile.get('sparql')):
            if sparql and IS_URI.match(sparql):
                triples.append(f'{iri_formatted} dcterms:endpointURL <{sparql}>')

        for con in _flatten_and_stringify(profile.get('con')):
            if con and IS_URI.match(con):
                triples.append(f'{iri_formatted} owl:sameAs <{con}>')

        # Add identifier and category (use the extracted string, not the original raw_id)
        escaped_raw_id = _escape_sparql_literal(raw_id_str)
        triples.append(f'{iri_formatted} dcterms:identifier "{escaped_raw_id}"')

        escaped_category = _escape_sparql_literal(str(category))
        triples.append(f'{iri_formatted} dcat:theme "{escaped_category}"')

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

        await _update_query(query_main)
        logger.info(f"Successfully inserted main profile data for IRI: {iri}")

    except Exception as error:
        logger.error(f"Cannot insert main profile data with iri: {iri}. Error: {error}")
        return

    # Insert vocabulary and keyword data
    try:
        vocab_triples = ""
        for voc in _flatten_and_stringify(profile.get('voc')):
            if voc and IS_URI.match(voc):
                # Preserve original URI form
                vocab_triples += f"{iri_formatted} void:vocabulary <{voc}> .\n"

        keyword_triples = ""
        for tag in _flatten_and_stringify(profile.get('tags')):
            if tag:
                escaped_tag = _escape_sparql_literal(tag)
                keyword_triples += f'{iri_formatted} dcat:keyword "{escaped_tag}" .\n'

        if vocab_triples or keyword_triples:
            query_vocab = f"""
                PREFIX void: <http://rdfs.org/ns/void#>
                PREFIX dcat: <http://www.w3.org/ns/dcat#>
                INSERT DATA {{
                {vocab_triples.rstrip()}
                {keyword_triples.rstrip()}
            }}
            """.strip()

            await _update_query(query_vocab)
            logger.info(f"Successfully inserted vocabulary and keyword data for IRI: {iri}")

    except Exception as error:
        logger.warning(f"Cannot insert vocabulary or keyword data for IRI: {iri}. Error: {error}")

    # Insert subject data
    try:
        subject_triples = ""
        for subj in _flatten_and_stringify(profile.get('sbj')):
            if subj and IS_URI.match(subj):
                # Preserve original URI form
                subject_triples += f'{iri_formatted} dcterms:subject <{subj}> .\n'

        if subject_triples:
            query_subject = f"""
            PREFIX dcterms: <http://purl.org/dc/terms/>
            INSERT DATA {{
            {subject_triples.rstrip()}
            }}
            """.strip()

            await _update_query(query_subject)
            logger.info(f"Successfully inserted subject data for IRI: {iri}")

    except Exception as error:
        logger.warning(f"Cannot insert subject data for IRI: {iri}. Error: {error}")


if __name__ == '__main__':
    # Ensure predictor is loaded before running
    load_predictor()
    asyncio.run(generate_profile_from_store())