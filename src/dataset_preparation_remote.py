import asyncio
import logging
import sys
import xml.etree.ElementTree as eT
from typing import Any

import aiohttp
import pandas as pd

from src.util import is_voc_allowed, is_curi_allowed

MAX_OFFSET = 1000
ENDPOINT_TIMEOUT = 600

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset_preparation_remote")


async def _fetch_query(session: aiohttp.ClientSession, endpoint: str, query: str, timeout: int) -> str:
    async with session.post(endpoint, data={"query": query}, timeout=timeout) as response:
        return await response.text()


async def async_select_remote_vocabularies(
    endpoint: str,
    timeout: int = 300,
    filter_voc: bool = True
) -> list[str]:
    """
    Retrieves distinct predicate‐based vocabulary URIs from the endpoint.
    If filter_voc=False, skips is_voc_allowed() filtering.
    """
    logger.info(f"[VOC] Starting vocabulary query for endpoint: {endpoint}")
    vocabularies: set[str] = set()

    query = """
        SELECT DISTINCT ?predicate
        WHERE {
            ?subject ?predicate ?object.
        }
        LIMIT 1000
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="predicate"]/sparql:uri', ns)

            if not bindings:
                logger.debug(f"[VOC] No predicate bindings found at endpoint {endpoint}.")
            for binding in bindings:
                predicate_uri = binding.text or ""
                if "#" in predicate_uri:
                    vocabulary_uri = predicate_uri.split("#")[0]
                elif "/" in predicate_uri:
                    parts = predicate_uri.rstrip("/").split("/")
                    vocabulary_uri = "/".join(parts[:-1]) if len(parts) > 1 else predicate_uri
                else:
                    vocabulary_uri = predicate_uri

                if not vocabulary_uri:
                    continue

                if filter_voc:
                    if is_voc_allowed(vocabulary_uri):
                        vocabularies.add(vocabulary_uri)
                else:
                    vocabularies.add(vocabulary_uri)

        except Exception as e:
            logger.warning(f"[VOC] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[VOC] Finished vocabulary query for endpoint: {endpoint} (found {len(vocabularies)} vocabularies)")
    return list(vocabularies)


async def async_select_remote_class(
    endpoint: str,
    timeout: int = 300,
    filter_curi: bool = True
) -> list[str]:
    """
    Retrieves distinct RDF classes and their instance counts.
    If filter_curi=False, skips is_curi_allowed() filtering.
    """
    logger.info(f"[CLS] Starting class query for endpoint: {endpoint}")
    classes: list[str] = []

    query = """
        SELECT ?class (COUNT(?instance) AS ?instanceCount)
        WHERE {
            ?instance a ?class .
        }
        GROUP BY ?class
        ORDER BY DESC(?instanceCount)
        LIMIT 1000
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="class"]/sparql:uri', ns)

            if not bindings:
                logger.debug(f"[CLS] No class bindings found at endpoint {endpoint}.")
            for binding in bindings:
                class_uri = binding.text or ""
                if not class_uri:
                    continue

                if filter_curi:
                    if is_curi_allowed(class_uri):
                        classes.append(class_uri)
                else:
                    classes.append(class_uri)

        except Exception as e:
            logger.warning(f"[CLS] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[CLS] Finished class query for endpoint: {endpoint} (found {len(classes)} classes)")
    return classes


async def async_select_remote_connection(endpoint: str, timeout: int = 300) -> list[str]:
    """
    Retrieves distinct objects of owl:sameAs links.
    """
    logger.info(f"[CON] Starting connection query for endpoint: {endpoint}")
    connections: list[str] = []

    query = """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT DISTINCT ?o
        WHERE {
            ?s owl:sameAs ?o .
        }
        LIMIT 1000
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="o"]/sparql:uri', ns)

            if not bindings:
                logger.debug(f"[CON] No connection bindings found at endpoint {endpoint}.")
            for binding in bindings:
                obj_uri = binding.text or ""
                if obj_uri:
                    connections.append(obj_uri)

        except Exception as e:
            logger.warning(f"[CON] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[CON] Finished connection query for endpoint: {endpoint} (found {len(connections)} connections)")
    return connections


async def async_select_remote_label(
    endpoint: str,
    timeout: int = 300,
    en: bool = True
) -> list[str]:
    """
    Retrieves distinct labels (rdfs:label, foaf:name, etc.) with optional English‐only filter.
    If no labels found with skosxl + others, falls back to rdfs:label only.
    """
    logger.info(f"[LAB] Starting label query for endpoint: {endpoint} (en={en})")
    labels: list[str] = []
    ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}

    query = """
        PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX awol: <http://bblfish.net/work/atom-owl/2006-06-06/#>
        PREFIX wdrs: <http://www.w3.org/2007/05/powder-s#>
        PREFIX schema: <http://schema.org/>
        SELECT DISTINCT ?o
        WHERE {
            ?s a ?label .
            { ?s rdfs:label ?o } UNION
            { ?s foaf:name ?o } UNION
            { ?s skos:prefLabel ?o } UNION
            { ?s rdfs:comment ?o } UNION
            { ?s awol:label ?o } UNION
            { ?s skos:note ?o } UNION
            { ?s wdrs:text ?o } UNION
            { ?s skosxl:prefLabel ?o } UNION
            { ?s skosxl:literalForm ?o } UNION
            { ?s schema:name ?o }
    """
    if en:
        query += 'FILTER(langMatches(lang(?o), "en")) '
    query += "} LIMIT 1000"

    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            bindings = root.findall('.//sparql:binding[@name="o"]/sparql:literal', ns)

            if not bindings:
                logger.debug(f"[LAB] No label bindings (primary) at endpoint {endpoint}. Trying fallback.")
                query_fallback = """
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    SELECT DISTINCT ?o
                    WHERE {
                        ?s rdfs:label ?o
                """
                if en:
                    query_fallback += 'FILTER(langMatches(lang(?o), "en")) '
                query_fallback += "} LIMIT 1000"

                result_text = await _fetch_query(session, endpoint, query_fallback, timeout)
                root = eT.fromstring(result_text)
                bindings = root.findall('.//sparql:binding[@name="o"]/sparql:literal', ns)

            for binding in bindings or []:
                lit = binding.text or ""
                if lit:
                    labels.append(lit)

        except Exception as e:
            logger.warning(f"[LAB] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[LAB] Finished label query for endpoint: {endpoint} (found {len(labels)} labels)")
    return labels


async def async_select_remote_title(endpoint: str, timeout: int = 300) -> str:
    """
    Attempts to fetch a dcterms:title for the dataset. Returns empty string if none found.
    """
    logger.info(f"[TITLE] Starting title query for endpoint: {endpoint}")
    title = ""
    query = """
        PREFIX dcterms: <http://purl.org/dc/terms/>
        SELECT ?classUri
        WHERE {
            ?type dcterms:title ?classUri .
        }
        LIMIT 1
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="classUri"]/sparql:uri', ns)

            if not bindings:
                logger.debug(f"[TITLE] No title found at endpoint {endpoint}.")
            else:
                title = bindings[0].text or ""

        except Exception as e:
            logger.warning(f"[TITLE] Query execution error: {e}. Endpoint: {endpoint}")
            return ""

    logger.info(f"[TITLE] Finished title query for endpoint: {endpoint}")
    return title


async def async_select_remote_tld(endpoint: str, limit: int = 1000, timeout: int = 300) -> list[str]:
    """
    Retrieves distinct IRIs (?o) from the dataset and extracts their TLD.
    """
    logger.info(f"[TLDS] Starting TLD query for endpoint: {endpoint}")
    tlds: set[str] = set()

    query = f"""
        SELECT DISTINCT ?o
        WHERE {{
            ?s ?p ?o .
            FILTER(isIRI(?o))
        }}
        LIMIT {limit}
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="o"]/sparql:uri', ns)

            if not bindings:
                logger.debug(f"[TLDS] No TLD bindings found at endpoint {endpoint}.")
            for binding in bindings:
                url = binding.text or ""
                if url.lower().startswith(("http://", "https://")):
                    try:
                        host = url.split("/")[2]
                        tld = host.split(".")[-1]
                        if 1 < len(tld) <= 10:
                            tlds.add(tld)
                    except Exception as e:
                        logger.debug(f"[TLDS] Error parsing TLD for URL {url}: {e}")

        except Exception as e:
            logger.warning(f"[TLDS] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[TLDS] Finished TLD query for endpoint: {endpoint} (found {len(tlds)} TLDs)")
    return list(tlds)


async def async_select_remote_property(
    endpoint: str,
    timeout: int = 300,
    filter_voc: bool = True
) -> list[str]:
    """
    Retrieves distinct properties ordered by usage count.
    If filter_voc=False, skips is_voc_allowed() filtering.
    """
    logger.info(f"[PROP] Starting property query for endpoint: {endpoint}")
    properties: list[str] = []

    query = f"""
        SELECT ?property (COUNT(?s) AS ?usageCount)
        WHERE {{
            ?s ?property ?o .
            {'FILTER (?property != rdf:type)' if filter_voc else '# no rdf:type filter'}
        }}
        GROUP BY ?property
        ORDER BY DESC(?usageCount)
        LIMIT 1000
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="property"]/sparql:uri', ns)

            if not bindings:
                logger.debug(f"[PROP] No property bindings found at endpoint {endpoint}.")
            for binding in bindings:
                prop_uri = binding.text or ""
                if not prop_uri:
                    continue

                if filter_voc:
                    if is_voc_allowed(prop_uri):
                        properties.append(prop_uri)
                else:
                    properties.append(prop_uri)

        except Exception as e:
            logger.warning(f"[PROP] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[PROP] Finished property query for endpoint: {endpoint} (found {len(properties)} properties)")
    return properties


async def async_select_remote_property_names(
    endpoint: str,
    timeout: int = 300,
    filter_voc: bool = True
) -> list[str]:
    """
    Retrieves local names (after '#' or '/') of the most used properties.
    If filter_voc=False, skips is_voc_allowed() filtering.
    """
    logger.info(f"[PNAME] Starting property name query for endpoint: {endpoint}")
    local_property_names: list[str] = []
    processed: set[str] = set()

    query = f"""
        SELECT ?property (COUNT(?s) AS ?usageCount)
        WHERE {{
            ?s ?property ?o .
            {'FILTER (?property != rdf:type)' if filter_voc else '# no rdf:type filter'}
        }}
        GROUP BY ?property
        ORDER BY DESC(?usageCount)
        LIMIT 1000
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="property"]/sparql:uri', ns)

            if not bindings:
                logger.debug(f"[PNAME] No property name bindings found at endpoint {endpoint}.")
            for binding in bindings:
                prop_uri = binding.text or ""
                if not prop_uri:
                    continue

                if filter_voc and not is_voc_allowed(prop_uri):
                    continue

                if "#" in prop_uri:
                    local_name = prop_uri.split("#")[-1]
                elif "/" in prop_uri:
                    local_name = prop_uri.rstrip("/").split("/")[-1]
                else:
                    local_name = prop_uri

                if local_name and local_name not in processed:
                    local_property_names.append(local_name)
                    processed.add(local_name)

        except Exception as e:
            logger.warning(f"[PNAME] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[PNAME] Finished property name query for endpoint: {endpoint} (found {len(local_property_names)} names)")
    return local_property_names


async def async_select_remote_class_name(
    endpoint: str,
    timeout: int = 300,
    filter_curi: bool = True
) -> list[str]:
    """
    Retrieves local names of the most used classes.
    If filter_curi=False, skips is_curi_allowed() filtering.
    """
    logger.info(f"[CNAME] Starting class name query for endpoint: {endpoint}")
    local_names: list[str] = []

    query = """
        SELECT ?class (COUNT(?instance) AS ?instanceCount)
        WHERE {
            ?instance a ?class .
        }
        GROUP BY ?class
        ORDER BY DESC(?instanceCount)
        LIMIT 1000
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="class"]/sparql:uri', ns)

            if not bindings:
                logger.debug(f"[CNAME] No class name bindings found at endpoint {endpoint}.")
            for binding in bindings:
                class_uri = binding.text or ""
                if not class_uri:
                    continue

                if filter_curi and not is_curi_allowed(class_uri):
                    continue

                if "#" in class_uri:
                    local_name = class_uri.split("#")[-1]
                elif "/" in class_uri:
                    local_name = class_uri.rstrip("/").split("/")[-1]
                else:
                    local_name = class_uri

                if local_name:
                    local_names.append(local_name)

        except Exception as e:
            logger.warning(f"[CNAME] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[CNAME] Finished class name query for endpoint: {endpoint} (found {len(local_names)} names)")
    return local_names


async def async_has_void_file(endpoint: str, timeout: int = 300) -> str | bool:
    """
    Checks whether a VOID dataset description exists for this endpoint.
    Returns the VOID URI if found, False otherwise.
    """
    logger.info(f"[VOID] Checking for VOID file at endpoint: {endpoint}")
    query = f"""
        PREFIX void: <http://rdfs.org/ns/void#>
        SELECT DISTINCT ?s
        WHERE {{
            ?s a void:Dataset ;
               void:sparqlEndpoint <{endpoint}> .
        }}
        LIMIT 1
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="s"]/sparql:uri', ns)

            for binding in bindings:
                uri = binding.text or ""
                if uri:
                    logger.info(f"[VOID] VOID file found: {uri}")
                    return uri
            return False

        except Exception as e:
            logger.warning(f"[VOID] Error checking for VOID file: {e}. Endpoint: {endpoint}")
            return False


async def async_select_void_description(
    endpoint: str,
    timeout: int = 300,
    void_file: bool = False
) -> list[str]:
    """
    Retrieves dcterms:description from a VOID file. If none found and void_file=False,
    tries to locate a VOID file and rerun.
    """
    logger.info(f"[VDESC] Starting VOID description query for endpoint: {endpoint}")
    descriptions: set[str] = set()

    query = """
        PREFIX dcterms: <http://purl.org/dc/terms/>
        SELECT ?desc
        WHERE {
            ?s dcterms:description ?desc .
        }
        LIMIT 1
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="desc"]/*', ns)

            for binding in bindings:
                desc_text = binding.text or ""
                if desc_text:
                    descriptions.add(desc_text)

            if not descriptions and not void_file:
                void_uri = await async_has_void_file(endpoint, timeout)
                if void_uri:
                    return await async_select_void_description(void_uri, timeout, True)

        except Exception as e:
            logger.warning(f"[VDESC] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[VDESC] Finished VOID description query for endpoint: {endpoint}")
    return list(descriptions)


async def async_select_void_license(
    endpoint: str,
    timeout: int = 300,
    void_file: bool = False
) -> list[str]:
    """
    Retrieves dcterms:license from a VOID file. If none found and void_file=False,
    tries to locate a VOID file and rerun.
    """
    logger.info(f"[VLIC] Starting VOID license query for endpoint: {endpoint}")
    licenses: set[str] = set()

    query = """
        PREFIX dcterms: <http://purl.org/dc/terms/>
        SELECT ?desc
        WHERE {
            ?s dcterms:license ?desc .
        }
        LIMIT 1
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="desc"]/*', ns)

            for binding in bindings:
                lic_text = binding.text or ""
                if lic_text:
                    licenses.add(lic_text)

            if not licenses and not void_file:
                void_uri = await async_has_void_file(endpoint, timeout)
                if void_uri:
                    return await async_select_void_license(void_uri, timeout, True)

        except Exception as e:
            logger.warning(f"[VLIC] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[VLIC] Finished VOID license query for endpoint: {endpoint}")
    return list(licenses)


async def async_select_void_creator(
    endpoint: str,
    timeout: int = 300,
    void_file: bool = False
) -> list[str]:
    """
    Retrieves dcterms:creator from a VOID file. If none found and void_file=False,
    tries to locate a VOID file and rerun.
    """
    logger.info(f"[VCRE] Starting VOID creator query for endpoint: {endpoint}")
    creators: set[str] = set()

    query = """
        PREFIX dcterms: <http://purl.org/dc/terms/>
        SELECT ?desc
        WHERE {
            ?s dcterms:creator ?desc .
        }
        LIMIT 1
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="desc"]/*', ns)

            for binding in bindings:
                cre_text = binding.text or ""
                if cre_text:
                    creators.add(cre_text)

            if not creators and not void_file:
                void_uri = await async_has_void_file(endpoint, timeout)
                if void_uri:
                    return await async_select_void_creator(void_uri, timeout, True)

        except Exception as e:
            logger.warning(f"[VCRE] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    logger.info(f"[VCRE] Finished VOID creator query for endpoint: {endpoint}")
    return list(creators)


async def async_select_void_subject_remote(
    endpoint: str,
    timeout: int = 300,
    void_file: bool = False
) -> list[str]:
    """
    Retrieves subjects of type void:Dataset, then fetches dcterms:subject for each.
    """
    logger.info(f"[VSUBJ] Starting VOID subject query for endpoint: {endpoint}")
    dataset_uris: set[str] = set()

    query = """
        PREFIX void: <http://rdfs.org/ns/void#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?s
        WHERE {
            ?s rdf:type void:Dataset .
        }
        LIMIT 100
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
            bindings = root.findall('.//sparql:binding[@name="s"]/sparql:uri', ns)

            for binding in bindings:
                uri = binding.text or ""
                if uri:
                    dataset_uris.add(uri)

            if not dataset_uris and not void_file:
                void_uri = await async_has_void_file(endpoint, timeout)
                if void_uri:
                    return await async_select_void_subject_remote(void_uri, timeout, True)

        except Exception as e:
            logger.warning(f"[VSUBJ] Query execution error: {e}. Endpoint: {endpoint}")
            return []

    class_names: set[str] = set()
    async with aiohttp.ClientSession() as session:
        for ds_uri in dataset_uris:
            query2 = f"""
                PREFIX dcterms: <http://purl.org/dc/terms/>
                SELECT DISTINCT ?classUri
                WHERE {{
                    <{ds_uri}> dcterms:subject ?classUri .
                }}
                LIMIT 100
            """
            try:
                result_text = await _fetch_query(session, ds_uri, query2, timeout)
                root = eT.fromstring(result_text)
                ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}
                bindings2 = root.findall('.//sparql:binding[@name="classUri"]/sparql:uri', ns)

                for binding in bindings2:
                    cn = binding.text or ""
                    if cn:
                        class_names.add(cn)

            except Exception as e:
                logger.warning(f"[VSUBJ] Error processing VOID subjects for {ds_uri}: {e}")

    logger.info(f"[VSUBJ] Finished VOID subject query for endpoint: {endpoint}")
    return list(class_names)


async def process_endpoint(
    row: pd.Series,
    filter_curi: bool = True,
    filter_voc: bool = True
) -> list[Any]:
    """
    Orchestrates all remote queries for one endpoint row:
     - title, vocabularies, class URIs, property URIs, class names, property names, labels, TLDs, creator, license, connections.
    Respects filter_curi / filter_voc flags to disable respective filtering.
    Returns a list of results matching the columns in main_normal().
    """
    endpoint = str(row["sparql_url"])
    row_id = str(row["id"])
    logger.info(f"[PROC] Processing endpoint {row_id}")

    tasks = {
        "title": async_select_remote_title(endpoint),
        "voc": async_select_remote_vocabularies(endpoint, filter_voc=filter_voc),
        "curi": async_select_remote_class(endpoint, filter_curi=filter_curi),
        "puri": async_select_remote_property(endpoint, filter_voc=filter_voc),
        "lcn": async_select_remote_class_name(endpoint, filter_curi=filter_curi),
        "lpn": async_select_remote_property_names(endpoint, filter_voc=filter_voc),
        "lab": async_select_remote_label(endpoint),
        "tlds": async_select_remote_tld(endpoint),
        "creator": async_select_void_creator(endpoint),
        "license": async_select_void_license(endpoint),
        "con": async_select_remote_connection(endpoint),
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    result_dict = dict(zip(tasks.keys(), results))

    logger.info(f"[PROC] Finished processing endpoint {row_id}")
    return [
        row_id,
        result_dict.get("title") or "",
        result_dict.get("voc") or [],
        result_dict.get("curi") or [],
        result_dict.get("puri") or [],
        result_dict.get("lcn") or [],
        result_dict.get("lpn") or [],
        result_dict.get("lab") or [],
        result_dict.get("tlds") or [],
        endpoint,
        result_dict.get("creator") or [],
        result_dict.get("license") or [],
        result_dict.get("con") or [],
        str(row["category"]),
    ]


async def process_endpoint_void(row: pd.Series) -> list[Any]:
    """
    Orchestrates VOID‐related queries for one endpoint row:
     - subject & description from VOID.
    """
    endpoint = str(row["sparql_url"])
    row_id = str(row["id"])
    logger.info(f"[VOID-PROC] Processing VOID endpoint {row_id}")

    tasks = {
        "sbj": async_select_void_subject_remote(endpoint),
        "dsc": async_select_void_description(endpoint),
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    result_dict = dict(zip(tasks.keys(), results))

    logger.info(f"[VOID-PROC] Finished processing VOID endpoint {row_id}")
    return [
        row_id,
        result_dict.get("sbj") or [],
        result_dict.get("dsc") or [],
        str(row["category"]),
    ]


async def process_endpoint_full_inplace(endpoint: str) -> dict[str, Any]:
    """
    Combines both ‘process_endpoint’ and ‘process_endpoint_void’ into a single record.
    Returns a dict with all relevant fields keyed by their names.
    """
    # Construct a temporary row-like object for queries
    row = pd.Series({"id": "", "sparql_url": endpoint, "category": ""})

    void_uri = await async_has_void_file(endpoint)
    if void_uri:
        title = await async_select_remote_title(void_uri)
    else:
        title = await async_select_remote_title(endpoint)
    if not title:
        title = endpoint

    data_list = await process_endpoint(row)
    void_list = await process_endpoint_void(row)

    return {
        "id": endpoint,
        "title": title,
        "sbj": void_list[1],
        "dsc": void_list[2],
        "voc": data_list[2],
        "curi": data_list[3],
        "puri": data_list[4],
        "lcn": data_list[5],
        "lpn": data_list[6],
        "lab": data_list[7],
        "tlds": data_list[8],
        "sparql": endpoint,
        "creator": data_list[10],
        "license": data_list[11],
        "con": data_list[12],
    }


async def main_normal(
    filter_curi: bool = True,
    filter_voc: bool = True
) -> None:
    """
    Main entrypoint for “normal” remote dataset processing.
    If filter_curi=False, skips is_curi_allowed() checks.
    If filter_voc=False, skips is_voc_allowed() checks.
    """
    logger.info("[MAIN] Starting asynchronous remote dataset processing (normal mode).")

    try:
        lod_frame = pd.read_csv("../data/raw/sparql_full_download.csv")
        lod_frame = lod_frame[~lod_frame["category"].str.strip().isin(["user_generated"])]
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        sys.exit(1)

    lod_frame = lod_frame.drop_duplicates(subset=["sparql_url"])
    lod_frame = lod_frame[lod_frame["sparql_url"].notna() & (lod_frame["sparql_url"] != "")]

    tasks = [
        asyncio.wait_for(process_endpoint(row, filter_curi=filter_curi, filter_voc=filter_voc), timeout=ENDPOINT_TIMEOUT)
        for _, row in lod_frame.iterrows()
    ]
    total = len(tasks)
    logger.info(f"[MAIN] Total endpoints to process: {total}")

    results: list[list[Any]] = []
    processed = 0

    for coro in asyncio.as_completed(tasks):
        try:
            res = await coro
            if res is not None:
                results.append(res)
        except asyncio.TimeoutError:
            logger.warning("[MAIN] Timeout processing an endpoint.")
        except Exception as e:
            logger.warning(f"[MAIN] Error processing an endpoint: {e}")
        processed += 1
        logger.info(f"[MAIN] Processed {processed}/{total} endpoints")

    df = pd.DataFrame(
        results,
        columns=[
            "id",
            "title",
            "voc",
            "curi",
            "puri",
            "lcn",
            "lpn",
            "lab",
            "tlds",
            "sparql",
            "creator",
            "license",
            "con",
            "category",
        ],
    )
    output_path = "../data/raw/remote/remote_feature_set_sparqlwrapper.json"
    df.to_json(output_path, orient="records")
    logger.info(f"[MAIN] Finished processing. Output saved to {output_path}")


async def main_void(
    filter_curi: bool = True,
    filter_voc: bool = True
) -> None:
    """
    Main entrypoint for VOID dataset processing.
    The filter flags are accepted for API symmetry, but VOID queries do not use them.
    """
    logger.info("[VOID-MAIN] Starting asynchronous VOID dataset processing.")

    try:
        lod_frame = pd.read_csv("../data/raw/sparql_full_download.csv")
        lod_frame = lod_frame[~lod_frame["category"].str.strip().isin(["user_generated"])]
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        sys.exit(1)

    lod_frame = lod_frame.drop_duplicates(subset=["sparql_url"])
    lod_frame = lod_frame[lod_frame["sparql_url"].notna() & (lod_frame["sparql_url"] != "")]

    tasks = [
        asyncio.wait_for(process_endpoint_void(row), timeout=ENDPOINT_TIMEOUT)
        for _, row in lod_frame.iterrows()
    ]
    total = len(tasks)
    logger.info(f"[VOID-MAIN] Total VOID endpoints to process: {total}")

    results: list[list[Any]] = []
    processed = 0

    for coro in asyncio.as_completed(tasks):
        try:
            res = await coro
            if res is not None:
                results.append(res)
        except asyncio.TimeoutError:
            logger.warning("[VOID-MAIN] Timeout processing a VOID endpoint.")
        except Exception as e:
            logger.warning(f"[VOID-MAIN] Error processing a VOID endpoint: {e}")
        processed += 1
        logger.info(f"[VOID-MAIN] Processed {processed}/{total} VOID endpoints")

    df = pd.DataFrame(results, columns=["id", "sbj", "dsc", "category"])
    output_path = "../data/raw/remote/remote_void_feature_set_sparqlwrapper.json"
    df.to_json(output_path, orient="records")
    logger.info(f"[VOID-MAIN] Finished VOID processing. Output saved to {output_path}")


if __name__ == "__main__":
    # Run both normal and void processing with filters enabled by default
    asyncio.run(main_normal(filter_curi=True, filter_voc=True))
    asyncio.run(main_void(filter_curi=True, filter_voc=True))
