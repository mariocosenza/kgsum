import asyncio
import logging
import sys
import xml.etree.ElementTree as eT

import aiohttp
import pandas as pd

MAX_OFFSET = 1000
ENDPOINT_TIMEOUT = 600

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset_preparation_remote")


async def _fetch_query(session, endpoint, query, timeout):
    async with session.post(endpoint, data={'query': query}, timeout=timeout) as response:
        return await response.text()


async def async_select_remote_vocabularies(endpoint, timeout=300):
    logger.info(f"[VOC] Starting vocabulary query for endpoint: {endpoint}")
    vocabularies = set()
    offset = 0
    async with aiohttp.ClientSession() as session:
            query = """
            SELECT DISTINCT ?predicate
            WHERE {
                ?subject ?predicate ?object .
            }
            LIMIT 1000
            """
            try:
                result_text = await _fetch_query(session, endpoint, query, timeout)
                root = eT.fromstring(result_text)
                ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
                bindings = root.findall('.//sparql:binding[@name="predicate"]/sparql:uri', ns)
                if not bindings:
                    logger.debug(f"[VOC] No predicate bindings found at offset {offset}.")
                for binding in bindings:
                    predicate_uri = binding.text
                    if predicate_uri:
                        if "#" in predicate_uri:
                            vocabulary_uri = predicate_uri.split("#")[0]
                        elif "/" in predicate_uri:
                            parts = predicate_uri.split("/")
                            vocabulary_uri = "/".join(parts[:-1]) if len(parts) > 1 else predicate_uri
                        else:
                            vocabulary_uri = predicate_uri
                        if vocabulary_uri and not vocabulary_uri.startswith("http://www.w3.org/"):
                            vocabularies.add(vocabulary_uri)
                offset += 100
            except Exception as e:
                logger.warning(f"[VOC] Query execution error: {e}. Endpoint: {endpoint}")
                return ''
    logger.info(f"[VOC] Finished vocabulary query for endpoint: {endpoint} (found {len(vocabularies)} vocabularies)")
    return vocabularies


async def async_select_remote_class(endpoint, timeout=300):
    logger.info(f"[CLS] Starting class query for endpoint: {endpoint}")
    classes = []
    offset = 0
    async with aiohttp.ClientSession() as session:
        query = """
            SELECT ?class (COUNT(?instance) AS ?instanceCount)
            WHERE {
                ?instance a ?class .
            }
            GROUP BY ?class
            ORDER BY DESC(?instanceCount)
            LIMIT 1000
            """
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="class"]/sparql:uri', ns)
            if not bindings:
                logger.debug(f"[CLS] No class bindings found at offset {offset}.")
            for binding in bindings:
                classes.append(binding.text)
        except Exception as e:
            logger.warning(f"[CLS] Query execution error: {e}. Endpoint: {endpoint}")
            return ''
    logger.info(f"[CLS] Finished class query for endpoint: {endpoint} (found {len(classes)} classes)")
    return classes


async def async_select_remote_label(endpoint, limit=1000, timeout=300):
    logger.info(f"[LAB] Starting label query for endpoint: {endpoint}")
    labels = []
    offset = 0
    async with aiohttp.ClientSession() as session:
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label
            WHERE {{
                ?item rdfs:label ?label .
                FILTER(langMatches(lang(?label), "en"))
            }}
            LIMIT {limit} 
            """
            bindings = []
            try:
                result_text = await _fetch_query(session, endpoint, query, timeout)
                root = eT.fromstring(result_text)
                ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
                bindings = root.findall('.//sparql:binding[@name="label"]/sparql:literal', ns)
            except Exception as e:
                logger.debug(f"[LAB] No label bindings found at offset {offset}. Exception: {e}")

            try:
                if not bindings:
                    logger.debug(f"[LAB] No label bindings found at offset {offset} with filter; trying fallback.")
                    query = f"""
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    SELECT ?label
                    WHERE {{
                        ?item rdfs:label ?label .
                    }}
                    LIMIT {limit} 
                    """
                    result_text = await _fetch_query(session, endpoint, query, timeout)
                    root = eT.fromstring(result_text)
                    bindings = root.findall('.//sparql:binding[@name="label"]/sparql:literal', ns)
                if not bindings:
                    logger.debug(f"[LAB] No label bindings found at offset {offset}.")
                for binding in bindings:
                    labels.append(binding.text)
            except Exception as e:
                logger.warning(f"[LAB] Query execution error: {e}. Endpoint: {endpoint}")
                return ''
    logger.info(f"[LAB] Finished label query for endpoint: {endpoint} (found {len(labels)} labels)")
    return labels


async def async_select_remote_title(endpoint, timeout=300):
    logger.info(f"[TITLE] Starting class query for endpoint: {endpoint}")
    title = ''
    offset = 0
    async with aiohttp.ClientSession() as session:
        query = f"""
               PREFIX dcterms: <http://purl.org/dc/terms/> 
               SELECT ?classUri
               WHERE {{
                  ?type dcterms:title ?classUri .
               }} LIMIT 1
               """
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="classUri"]/sparql:uri', ns)
            if not bindings:
                logger.debug(f"[TITLE] No class bindings found at offset {offset}.")
            else:
                for binding in bindings:
                    title = binding.text
        except Exception as e:
            logger.warning(f"[TITLE] Query execution error: {e}. Endpoint: {endpoint}")
            return title
    logger.info(f"[TITLE] Finished class query for endpoint: {endpoint} (found {len(title)} title)")
    return title


async def async_select_remote_tld(endpoint, limit=1000, timeout=300):
    logger.info(f"[TLD] Starting TLD query for endpoint: {endpoint}")
    tlds = set()
    offset = 0
    async with aiohttp.ClientSession() as session:
        query = f"""
            SELECT DISTINCT ?o
            WHERE {{
                ?s ?p ?o .
                FILTER(isIRI(?o))
            }}
            LIMIT {limit} 
            """
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="o"]/sparql:uri', ns)
            if not bindings:
                logger.debug(f"[TLD] No TLD bindings found at offset {offset}.")
            for binding in bindings:
                url = binding.text
                if url and (url.startswith('http') or url.startswith('https')):
                    try:
                        tld = url.split('/')[2].split('.')[-1]
                        if 1 < len(tld) <= 10:
                            tlds.add(tld)
                    except Exception as e:
                        logger.debug(f"[TLD] Error parsing TLD for URL {url}: {e}")
            offset += 100
        except Exception as e:
            logger.warning(f"[TLD] Query execution error: {e}. Endpoint: {endpoint}")
            return ''
    logger.info(f"[TLD] Finished TLD query for endpoint: {endpoint} (found {len(tlds)} TLDs)")
    return tlds


async def async_select_remote_property(endpoint, timeout=300):
    logger.info(f"[PROP] Starting property query for endpoint: {endpoint}")
    properties = []
    offset = 0
    async with aiohttp.ClientSession() as session:
        query = """
        SELECT ?property (COUNT(?s) AS ?usageCount)
        WHERE {{
            ?s ?property ?o .
  
            # Optional: Filter out rdf:type if you want to exclude it
            # FILTER (?property != rdf:type)
         }}
        GROUP BY ?property
        ORDER BY DESC(?usageCount)
        LIMIT 1000
        """
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="property"]/sparql:uri', ns)
            if not bindings:
                logger.debug(f"[PROP] No property bindings found at offset {offset}.")
            for binding in bindings:
                properties.append(binding.text)
            offset += 100
        except Exception as e:
            logger.warning(f"[PROP] Query execution error: {e}. Endpoint: {endpoint}")
            return ''
    logger.info(f"[PROP] Finished property query for endpoint: {endpoint} (found {len(properties)} properties)")
    return properties


async def async_select_remote_property_names(endpoint, timeout=300):
    logger.info(f"[PNAME] Starting property name query for endpoint: {endpoint}")
    local_property_names = []
    processed = set()
    offset = 0
    async with aiohttp.ClientSession() as session:
        query =  """
        SELECT ?property (COUNT(?s) AS ?usageCount)
        WHERE {{
            ?s ?property ?o .
  
            # Optional: Filter out rdf:type if you want to exclude it
            # FILTER (?property != rdf:type)
         }}
        GROUP BY ?property
        ORDER BY DESC(?usageCount)
        LIMIT 1000
        """
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="property"]/sparql:uri', ns)
            if not bindings:
                logger.debug(f"[PNAME] No property name bindings found at offset {offset}.")
            for binding in bindings:
                prop_uri = binding.text
                if not prop_uri:
                    continue
                if "#" in prop_uri:
                    local_name = prop_uri.split("#")[-1]
                elif "/" in prop_uri:
                    local_name = prop_uri.split("/")[-1]
                else:
                    local_name = prop_uri
                if local_name and local_name not in processed:
                    local_property_names.append(local_name)
                    processed.add(local_name)
            offset += 100
        except Exception as e:
            logger.warning(f"[PNAME] Query execution error: {e}. Endpoint: {endpoint}")
            return ''
    logger.info(
        f"[PNAME] Finished property name query for endpoint: {endpoint} (found {len(local_property_names)} names)")
    return local_property_names


async def async_select_remote_class_name(endpoint, timeout=300):
    logger.info(f"[CNAME] Starting class name query for endpoint: {endpoint}")
    local_names = []
    offset = 0
    async with aiohttp.ClientSession() as session:
        query = """
            SELECT ?class (COUNT(?instance) AS ?instanceCount)
            WHERE {
                ?instance a ?class .
            }
            GROUP BY ?class
            ORDER BY DESC(?instanceCount)
            LIMIT 1000
            """
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="classUri"]/sparql:uri', ns)
            if not bindings:
                logger.debug(f"[CNAME] No class name bindings found at offset {offset}.")
            for binding in bindings:
                class_uri = binding.text
                if not class_uri:
                    continue
                if "#" in class_uri:
                    local_name = class_uri.split("#")[-1]
                elif "/" in class_uri:
                    local_name = class_uri.split("/")[-1]
                else:
                    local_name = class_uri
                local_names.append(local_name)
            offset += 100
        except Exception as e:
            logger.warning(f"[CNAME] Query execution error: {e}. Endpoint: {endpoint}")
            return ''
    logger.info(f"[CNAME] Finished class name query for endpoint: {endpoint} (found {len(local_names)} names)")
    return local_names


async def async_has_void_file(endpoint, timeout=300):
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
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="s"]/sparql:uri', ns)
            if bindings:
                for binding in bindings:
                    logger.info(f"[VOID] VOID file found: {binding.text}")
                    return binding.text
            return False
        except Exception as e:
            logger.warning(f"[VOID] Error checking for VOID file at endpoint: {endpoint}: {e}")
            return False


async def async_select_void_description(endpoint, timeout=300, void_file=False):
    logger.info(f"[VDESC] Starting VOID description query for endpoint: {endpoint}")
    query = """
          PREFIX dcterms: <http://purl.org/dc/terms/> 
          SELECT ?desc WHERE {
              ?s dcterms:description ?desc .
          } LIMIT 1
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            descriptions = {binding.text for binding in root.findall('.//sparql:binding[@name="desc"]/*', ns)}
            if not descriptions and not void_file:
                uri = await async_has_void_file(endpoint, timeout)
                if uri:
                    return await async_select_void_subject_remote(uri, timeout, True)
            logger.info(f"[VDESC] Finished VOID description query for endpoint: {endpoint}")
            return list(descriptions)
        except Exception as e:
            logger.warning(f"[VDESC] Error in VOID description query: {e}")
            return []


async def async_select_void_license(endpoint, timeout=300, void_file=False):
    logger.info(f"[VDESC] Starting VOID description query for endpoint: {endpoint}")
    query = """
          PREFIX dcterms: <http://purl.org/dc/terms/> 
          SELECT ?desc WHERE {
              ?s dcterms:license ?desc .
          } LIMIT 1
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            descriptions = {binding.text for binding in root.findall('.//sparql:binding[@name="desc"]/*', ns)}
            if not descriptions and not void_file:
                uri = await async_has_void_file(endpoint, timeout)
                if uri:
                    return await async_select_void_subject_remote(uri, timeout, True)
            logger.info(f"[VDESC] Finished VOID description query for endpoint: {endpoint}")
            return list(descriptions)
        except Exception as e:
            logger.warning(f"[VDESC] Error in VOID description query: {e}")
            return []


async def async_select_void_creator(endpoint, timeout=300, void_file=False):
    logger.info(f"[VDESC] Starting VOID description query for endpoint: {endpoint}")
    query = """
          PREFIX dcterms: <http://purl.org/dc/terms/> 
          SELECT ?desc WHERE {
              ?s dcterms:creator ?desc .
          } LIMIT 1
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            descriptions = {binding.text for binding in root.findall('.//sparql:binding[@name="desc"]/*', ns)}
            if not descriptions and not void_file:
                uri = await async_has_void_file(endpoint, timeout)
                if uri:
                    return await async_select_void_subject_remote(uri, timeout, True)
            logger.info(f"[VDESC] Finished VOID description query for endpoint: {endpoint}")
            return list(descriptions)
        except Exception as e:
            logger.warning(f"[VDESC] Error in VOID description query: {e}")
            return []


async def async_select_void_subject_remote(endpoint, timeout=300, void_file=False):
    logger.info(f"[VSUBJ] Starting VOID subject query for endpoint: {endpoint}")
    query = """
        PREFIX void: <http://rdfs.org/ns/void#> 
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?s 
        WHERE {
            ?s rdf:type void:Dataset .
        } LIMIT 100
    """
    async with aiohttp.ClientSession() as session:
        try:
            result_text = await _fetch_query(session, endpoint, query, timeout)
            root = eT.fromstring(result_text)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            local_names = {binding.text for binding in root.findall('.//sparql:binding[@name="s"]/sparql:uri', ns)}
            if not local_names and not void_file:
                uri = await async_has_void_file(endpoint, timeout)
                if uri:
                    return await async_select_void_subject_remote(uri, timeout, True)
        except Exception as e:
            logger.warning(f"[VSUBJ] VOID subject query error for endpoint: {endpoint}: {e}")
            return []
        class_names = set()
        for local_name in local_names:
            query2 = f"""
                PREFIX dcterms: <http://purl.org/dc/terms/>
                SELECT DISTINCT ?classUri
                WHERE {{
                    <{local_name}> dcterms:subject ?classUri .
                }} LIMIT 100
            """
            try:
                result_text = await _fetch_query(session, endpoint, query2, timeout)
                root = eT.fromstring(result_text)
                ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
                bindings = root.findall('.//sparql:binding[@name="classUri"]/sparql:uri', ns)
                for binding in bindings:
                    class_names.add(binding.text)
            except Exception as e:
                logger.warning(f"[VSUBJ] Error processing VOID subjects for {local_name} at endpoint {endpoint}: {e}")
        logger.info(f"[VSUBJ] Finished VOID subject query for endpoint: {endpoint}")
        return list(class_names)


async def process_endpoint(row):
    endpoint = row['sparql_url']
    logger.info(f"[PROC] Processing endpoint {row['id']}")
    tasks = {
        'title': async_select_remote_title(endpoint),
        'voc': async_select_remote_vocabularies(endpoint),
        'curi': async_select_remote_class(endpoint),
        'puri': async_select_remote_property(endpoint),
        'lcn': async_select_remote_class_name(endpoint),
        'lpn': async_select_remote_property_names(endpoint),
        'lab': async_select_remote_label(endpoint),
        'tlds': async_select_remote_tld(endpoint),
        'creator': async_select_void_creator(endpoint),
        'license': async_select_void_license(endpoint)
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    result_dict = dict(zip(tasks.keys(), results))
    logger.info(f"[PROC] Finished processing endpoint {row['id']}")
    return [row['id'], result_dict.get('title'), result_dict.get('voc'), result_dict.get('curi'),
            result_dict.get('puri'), result_dict.get('lcn'), result_dict.get('lpn'),
            result_dict.get('lab'), result_dict.get('tlds'), row['sparql_url'], result_dict.get('creator'),
            result_dict.get('license'), row['category']]


async def process_endpoint_void(row):
    endpoint = row['sparql_url']
    logger.info(f"[VOID-PROC] Processing VOID endpoint {row['id']}")
    tasks = {
        'sbj': async_select_void_subject_remote(endpoint),
        'dsc': async_select_void_description(endpoint)
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    result_dict = dict(zip(tasks.keys(), results))
    logger.info(f"[VOID-PROC] Finished processing VOID endpoint {row['id']}")
    return [row['id'], result_dict.get('sbj'), result_dict.get('dsc'), row['category']]


async def main_normal():
    logger.info("[MAIN] Starting asynchronous remote dataset processing (normal mode).")
    try:
        lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        sys.exit(1)
    lod_frame = lod_frame.drop_duplicates(subset=['sparql_url'])
    tasks = [
        asyncio.wait_for(process_endpoint(row), timeout=ENDPOINT_TIMEOUT)
        for _, row in lod_frame.iterrows() if row['sparql_url']
    ]
    total = len(tasks)
    logger.info(f"[MAIN] Total endpoints to process: {total}")
    results = []
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
        columns=['id', 'title', 'voc', 'curi', 'puri', 'lcn', 'lpn', 'lab', 'tld', 'sparql', 'creator', 'license', 'category']
    )
    df.to_json('../data/raw/remote/remote_feature_set_sparqlwrapper.json', orient='records')
    logger.info("[MAIN] Finished asynchronous remote dataset processing (normal mode).")


async def main_void():
    logger.info("[VOID-MAIN] Starting asynchronous VOID dataset processing.")
    try:
        lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        sys.exit(1)
    lod_frame = lod_frame.drop_duplicates(subset=['sparql_url'])
    tasks = [
        asyncio.wait_for(process_endpoint_void(row), timeout=ENDPOINT_TIMEOUT)
        for _, row in lod_frame.iterrows() if row['sparql_url']
    ]
    total = len(tasks)
    logger.info(f"[VOID-MAIN] Total VOID endpoints to process: {total}")
    results = []
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
    df = pd.DataFrame(
        results,
        columns=['id', 'sbj', 'dsc', 'category']
    )
    df.to_json('../data/raw/remote/remote_void_feature_set_sparqlwrapper.json', orient='records')
    logger.info("[VOID-MAIN] Finished asynchronous VOID dataset processing.")


async def process_endpoint_full_inplace(endpoint) -> dict[str, set | str | None | list]:
    row = {'sparql_url': endpoint, 'id': '', 'category': ''}
    void_endpoint = await async_has_void_file(endpoint)
    if void_endpoint:
        title = await async_select_remote_title(void_endpoint)
    else:
        title = await async_select_remote_title(endpoint)
    if title == '':
        title = endpoint

    data = await process_endpoint(row)
    data_void = await process_endpoint_void(row)

    return {
        'id': endpoint,
        'title': title,
        'sbj': data_void[0],
        'dsc': data_void[1],
        'voc': data[1],
        'curi': data[2],
        'puri': data[3],
        'lcn': data[4],
        'lpn': data[5],
        'lab': data[6],
        'tlds': data[7],
        'sparql': endpoint,
        'creator': data[8],
        'license': data[9]
    }


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    if mode == "void":
        asyncio.run(main_void())
    else:
        asyncio.run(main_normal())
