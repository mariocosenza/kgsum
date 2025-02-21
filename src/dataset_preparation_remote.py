import queue
from concurrent.futures import ThreadPoolExecutor
from SPARQLWrapper import SPARQLWrapper
import xml.etree.ElementTree as eT
import pandas as pd
from conda.common.io import as_completed

from service.endpoint_lod import logger

MAX_OFFSET = 900


def select_remote_vocabularies_sparqlwrapper(endpoint, limit=100, timeout=300,
                                             max_offset=MAX_OFFSET):  # timeout increased to 300
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout)
    vocabularies = set()
    offset = 0

    while True:
        if offset >= max_offset:
            break

        sparql.setQuery(f"""
            SELECT DISTINCT ?predicate
            WHERE {{
                ?subject ?predicate ?object .
            }}
            ORDER BY ?predicate
            LIMIT {limit}
            OFFSET {offset}
        """)
        sparql.setReturnFormat('xml')

        try:
            results = sparql.query().response.read()
            root = eT.fromstring(results)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}

            bindings = root.findall('.//sparql:binding[@name="predicate"]/sparql:uri', ns)

            if not bindings:
                break

            for binding in bindings:
                print(binding.text)
                predicate_uri = binding.text
                if predicate_uri:
                    if "#" in predicate_uri:
                        vocabulary_uri = predicate_uri.split("#")[0]
                    elif "/" in predicate_uri:
                        vocabulary_uri = predicate_uri.split("/")
                        vocabulary_uri = "/".join(vocabulary_uri[:len(vocabulary_uri) - 1]) if len(
                            vocabulary_uri) > 1 else predicate_uri
                    elif 'HTML PUBLIC' in predicate_uri:
                        return ''
                    else:
                        vocabulary_uri = predicate_uri

                    if vocabulary_uri and not vocabulary_uri.startswith("http://www.w3.org/"):
                        vocabularies.add(vocabulary_uri)

            offset += 100  # offset increased by 100

        except Exception as e:
            logger.warning(f"Query execution error: {e}")
            return ''

    return vocabularies


def select_remote_class_sparqlwrapper(endpoint, limit=100, timeout=300,
                                      max_offset=MAX_OFFSET):  # timeout increased to 300
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout)
    classes = []
    offset = 0

    while True:
        if offset >= max_offset:
            break

        sparql.setQuery(f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?classUri
            WHERE {{
                ?classUri a ?type .
                FILTER (?type IN (rdfs:Class, owl:Class))
            }}
            ORDER BY ?classUri
            LIMIT {limit}
            OFFSET {offset}
        """)
        sparql.setReturnFormat('xml')

        try:
            results = sparql.query().response.read()
            root = eT.fromstring(results)

            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="classUri"]/sparql:uri', ns)

            if not bindings:
                break

            for binding in bindings:
                classes.append(binding.text)

            offset += 100  # offset increased by 100

        except Exception as e:
            logger.warning(f"Query execution error: {e}")
            return ''

    return classes


def select_remote_label_sparqlwrapper(endpoint, limit=100, timeout=300,
                                      max_offset=MAX_OFFSET):  # timeout increased to 300
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout)
    labels = []
    offset = 0

    while True:
        if offset >= max_offset:
            break

        sparql.setQuery(f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?type
            WHERE {{
                ?class rdfs:label ?type .
            }}
            ORDER BY ?type
            LIMIT {limit}
            OFFSET {offset}
        """)
        sparql.setReturnFormat('xml')

        try:
            results = sparql.query().response.read()
            root = eT.fromstring(results)

            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="type"]/sparql:literal', ns)

            if not bindings:
                break

            for binding in bindings:
                labels.append(binding.text)

            offset += 100  # offset increased by 100

        except Exception as e:
            logger.warning(f"Query execution error: {e}")
            return ''

    return labels


def select_remote_tld_sparqlwrapper(endpoint, limit=100, timeout=300,
                                    max_offset=MAX_OFFSET):  # timeout increased to 300
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout)
    tlds = set()
    offset = 0

    while True:
        if offset >= max_offset:
            break

        sparql.setQuery(f"""
            SELECT DISTINCT ?o
            WHERE {{
                ?s ?p ?o .
                FILTER(isIRI(?o))
            }}
            ORDER BY ?o
            LIMIT {limit}
            OFFSET {offset}
        """)
        sparql.setReturnFormat('xml')

        try:
            results = sparql.query().response.read()
            root = eT.fromstring(results)

            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="o"]/sparql:uri', ns)

            if not bindings:
                break

            for binding in bindings:
                url = binding.text
                if url and (url.startswith('http') or url.startswith('https')):
                    try:
                        tld = url.split('/')[2].split('.')[-1]
                        if 1 < len(tld) <= 10:
                            tlds.add(tld)
                    except:
                        pass

            offset += 100  # offset increased by 100

        except Exception as e:
            logger.warning(f"Query execution error: {e}")
            return ''

    return tlds


def select_remote_property_sparqlwrapper(endpoint, limit=100, timeout=300,
                                         max_offset=MAX_OFFSET):  # timeout increased to 300
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout)
    properties = []
    offset = 0

    while True:
        if offset >= max_offset:
            break

        sparql.setQuery(f"""
            SELECT DISTINCT ?property
            WHERE {{
                ?subject ?property ?object .
                FILTER isIRI(?property)
            }}
            ORDER BY ?property
            LIMIT {limit}
            OFFSET {offset}
        """)
        sparql.setReturnFormat('xml')

        try:
            results = sparql.query().response.read()
            root = eT.fromstring(results)

            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="property"]/sparql:uri', ns)

            if not bindings:
                break

            for binding in bindings:
                properties.append(binding.text)

            offset += 100  # offset increased by 100

        except Exception as e:
            logger.warning(f"Query execution error: {e}")
            return ''

    return properties


def select_remote_property_names_sparqlwrapper(endpoint, limit=100, timeout=300,
                                               max_offset=MAX_OFFSET):  # timeout increased to 300
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout)
    local_property_names = []
    processed_local_names = set()
    offset = 0

    while True:
        if offset >= max_offset:
            break

        sparql.setQuery(f"""
            SELECT DISTINCT ?property
            WHERE {{
                ?subject ?property ?object .
                FILTER isIRI(?property)
            }}
            ORDER BY ?property
            LIMIT {limit}
            OFFSET {offset}
        """)
        sparql.setReturnFormat('xml')

        try:
            results = sparql.query().response.read()
            root = eT.fromstring(results)

            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="property"]/sparql:uri', ns)

            if not bindings:
                break

            for binding in bindings:
                property_uri = binding.text
                if not property_uri:
                    continue

                if "#" in property_uri:
                    local_name = property_uri.split("#")[-1]
                elif "/" in property_uri:
                    local_name = property_uri.split("/")[-1]
                else:
                    local_name = property_uri

                if local_name and local_name not in processed_local_names:
                    local_property_names.append(local_name)
                    processed_local_names.add(local_name)

            offset += 100  # offset increased by 100

        except Exception as e:
            logger.warning(f"Query execution error: {e}")
            return ''

    return local_property_names


def select_remote_class_name_sparqlwrapper(endpoint, limit=10, timeout=300,
                                           max_offset=MAX_OFFSET):  # timeout increased to 300
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout)
    local_names = []
    offset = 0

    while True:
        if offset >= max_offset:
            break

        sparql.setQuery(f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT DISTINCT ?classUri
            WHERE {{
                ?s rdf:type ?classUri .
            }}
            ORDER BY ?classUri
            LIMIT {limit}
            OFFSET {offset}
        """)
        sparql.setReturnFormat('xml')

        try:
            results = sparql.query().response.read()
            root = eT.fromstring(results)

            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="classUri"]/sparql:uri', ns)

            if not bindings:
                break

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

            offset += 100  # offset increased by 100

        except Exception as e:
            logger.warning(f"Query execution error: {e.__cause__}")
            return ''

    return local_names


def _has_void_file(endpoint, timeout=300) -> bool | str:
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout)
    sparql.setReturnFormat('xml')
    sparql.setQuery(f"""
          PREFIX void: <http://rdfs.org/ns/void#>
          SELECT DISTINCT ?s
          WHERE {{ ?s a void:Dataset ; 
            void:sparqlEndpoint <{endpoint}> . }}
           LIMIT 1""")
    try:
        results = sparql.query().response.read()
        root = eT.fromstring(results)
        ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
        bindings = root.findall('.//sparql:binding[@name="s"]/sparql:uri', ns)
        if bindings:
            for binding in bindings:
                return binding.text
        return False
    except:
        return False





def select_void_description(endpoint, timeout=300, void_file=False) -> list:
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout)
    sparql.setReturnFormat('xml')
    sparql.setQuery("""
          PREFIX dcterms: <http://purl.org/dc/terms/> 

          SELECT ?desc WHERE {
              ?s dcterms:description ?desc .
          } LIMIT 1
          """)
    try:
        results = sparql.query().response.read()
        root = eT.fromstring(results)
        local_descriptions = set()
        ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
        bindings = root.findall('.//sparql:binding[@name="desc"]/*', ns)
        for binding in bindings:
            local_descriptions.add(binding.text)
        if not bindings and not void_file:
            uri = _has_void_file(endpoint)
            if uri:
                return select_void_subject_remote(uri, timeout, True)
        return list(local_descriptions)
    except Exception as e:
        print("Error:", e)
        return []


def select_void_subject_remote(endpoint, timeout=300, void_file=False) -> list:  # timeout increased to 300
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout)
    sparql.setReturnFormat('xml')
    sparql.setQuery("""
        PREFIX void: <http://rdfs.org/ns/void#> 
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?s 
        WHERE {
            ?s rdf:type void:Dataset .
    } LIMIT 100""")
    try:
        results = sparql.query().response.read()
        root = eT.fromstring(results)
        local_names = set()
        ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
        bindings = root.findall('.//sparql:binding[@name="s"]/sparql:uri', ns)
        for binding in bindings:
            local_names.add(binding.text)
        if not bindings and not void_file:
            uri = _has_void_file(endpoint)
            if uri:
                return select_void_subject_remote(uri, timeout, True)
    except:
        return []

    try:
        class_names = set()
        for local_name in local_names:
            sparql.setQuery(f"""
                PREFIX dcterms: <http://purl.org/dc/terms/>

                SELECT DISTINCT ?classUri
                WHERE {{
                    <{local_name}> dcterms:subject ?classUri .
                }} LIMIT 100""")
            results = sparql.query().response.read()
            root = eT.fromstring(results)
            ns = {'sparql': 'http://www.w3.org/2005/sparql-results#'}
            bindings = root.findall('.//sparql:binding[@name="classUri"]/sparql:uri', ns)
            for binding in bindings:
                class_names.add(binding.text)
        return list(class_names)
    except:
        return []


def process_row(row, index, result_queue):
    logger.info(f"Endpoint: {row['id']} Number: {index}")
    endpoint = row['sparql_url']
    try:
        # Define a dictionary mapping a label to its corresponding query function call.
        query_tasks = {
            'voc': (select_remote_vocabularies_sparqlwrapper, {'endpoint': endpoint, 'timeout': 300}),
            'curi': (select_remote_class_sparqlwrapper, {'endpoint': endpoint, 'timeout': 300}),
            'puri': (select_remote_property_sparqlwrapper, {'endpoint': endpoint, 'timeout': 300}),
            'lcn': (select_remote_class_name_sparqlwrapper, {'endpoint': endpoint, 'timeout': 300}),
            'lpn': (select_remote_property_names_sparqlwrapper, {'endpoint': endpoint, 'timeout': 300}),
            'lab': (select_remote_label_sparqlwrapper, {'endpoint': endpoint, 'timeout': 300}),
            'tld': (select_remote_tld_sparqlwrapper, {'endpoint': endpoint, 'timeout': 300}),
        }

        results_dict = {}

        with ThreadPoolExecutor(max_workers=7) as executor:
            future_to_key = {
                executor.submit(func, **params): key
                for key, (func, params) in query_tasks.items()
            }

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results_dict[key] = future.result()
                except Exception as e:
                    logger.warning(f"Error in task {key} for endpoint {row['id']}: {str(e)}")
                    results_dict[key] = ''  # or handle error appropriately

        # Prepare the aggregated results.
        result = [
            row['id'],
            results_dict.get('voc'),
            results_dict.get('curi'),
            results_dict.get('puri'),
            results_dict.get('lcn'),
            results_dict.get('lpn'),
            results_dict.get('lab'),
            results_dict.get('tld'),
            row['category']
        ]

        result_queue.put(result)
    except Exception as e:
        logger.warning(f"Error processing endpoint {row['id']}: {str(e)}")


def create_remote_dataset_sparqlwrapper():
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    lod_frame = lod_frame.drop_duplicates(subset=['sparql_url'])
    logger.info('Started dataset processing')

    result_queue = queue.Queue()

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create a future for each row that has a non-empty sparql_url.
        futures = [
            executor.submit(process_row, row, index, result_queue)
            for index, row in lod_frame.iterrows() if row['sparql_url']
        ]
        # Wait for all futures to complete.
        for future in futures:
            future.result()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    df = pd.DataFrame(
        results,
        columns=['id', 'voc', 'curi', 'puri', 'lcn', 'lpn', 'lab', 'tld', 'category']
    )

    df.to_json('../data/raw/remote/remote_feature_set_sparqlwrapper.json', orient='records')
    logger.info('Finished dataset processing')


def process_row_void(row, index, result_queue):
    logger.info(f"Endpoint: {row['id']} Number: {index}")
    endpoint = row['sparql_url']
    try:
        query_tasks = {
            'dsc': (select_void_description, {'endpoint': endpoint, 'timeout': 300}),
            'sbj': (select_void_subject_remote, {'endpoint': endpoint, 'timeout': 300})
        }

        results_dict = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_key = {
                executor.submit(func, **params): key
                for key, (func, params) in query_tasks.items()
            }

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results_dict[key] = future.result()
                except Exception as e:
                    logger.warning(f"Error in task {key} for endpoint {row['id']}: {str(e)}")
                    results_dict[key] = ''  # or handle error appropriately

        # Prepare the aggregated results.
        result = [
            row['id'],
            results_dict.get('sbj'),
            results_dict.get('dsc'),
            row['category']
        ]

        result_queue.put(result)
    except Exception as e:
        logger.warning(f"Error processing endpoint {row['id']}: {str(e)}")


def create_remote_dataset_sparqlwrapper_void():
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    lod_frame = lod_frame.drop_duplicates(subset=['sparql_url'])
    logger.info('Started dataset processing')

    result_queue = queue.Queue()

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create a future for each row that has a non-empty sparql_url.
        futures = [
            executor.submit(process_row_void, row, index, result_queue)
            for index, row in lod_frame.iterrows() if row['sparql_url']
        ]

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    df = pd.DataFrame(
        results,
        columns=['id', 'sbj', 'dsc', 'category']
    )

    df.to_json('../data/raw/remote/remote_void_feature_set_sparqlwrapper.json', orient='records')
    logger.info('Finished dataset processing')


create_remote_dataset_sparqlwrapper_void()
