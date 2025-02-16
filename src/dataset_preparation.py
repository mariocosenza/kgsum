import re
from os import listdir
import pandas as pd
import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib.term import _toPythonMapping
from rdflib import Graph

from service.endpoint_lod import logger


def force_string_converter(value):
    return str(value)
for dt in list(_toPythonMapping.keys()):
    _toPythonMapping[dt] = force_string_converter


categories = {
    'cross_domain', 'geography', 'government', 'life_sciences', 'linguistics', 'media', 'publications', 'social_networking', 'user_generated'
}

def select_local_vocabularies(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
        SELECT DISTINCT ?predicate
        WHERE {
            ?subject ?predicate ?object .
            FILTER (STRSTARTS(STR(?predicate), "http://"))
            FILTER (!STRSTARTS(STR(STRBEFORE(STR(?predicate), "#")), "http://www.w3.org/"))
        }
    """)
    vocabularies = []
    for row in qres:
        predicate_uri = str(row.predicate)
        if predicate_uri:
            vocabulary_uri = ""
            if "#" in predicate_uri:
                vocabulary_uri = predicate_uri.split("#")[0]
            elif "/" in predicate_uri:
                vocabulary_uri = predicate_uri.split("/")
                vocabulary_uri = "/".join(vocabulary_uri[:len(vocabulary_uri)-1]) if len(vocabulary_uri) > 1 else predicate_uri # Get path before last part
            else:
                vocabulary_uri = predicate_uri

            if vocabulary_uri and not vocabulary_uri.startswith("http://www.w3.org/"):
                vocabularies.append(vocabulary_uri)
    return vocabularies

def select_local_class(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""SELECT DISTINCT ?classUri
    WHERE {
            ?classUri a ?type .
            FILTER (?type IN (rdfs:Class, owl:Class))
    }""")
    classes = []
    for row in qres:
        classes.append(str(row.classUri))
    return classes

def select_local_label(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""SELECT DISTINCT ?type
    WHERE {
            ?class rdfs:label ?type .
    }""", initNs={"rdf": rdflib.RDF})
    labels = []
    for row in qres:
        labels.append(str(row.type))
    return labels


def select_local_tld(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
    SELECT DISTINCT ?o
    WHERE {
         ?s ?p ?o .
        FILTER(isIRI(?o))
    }
    """)
    tlds = set()
    for row in qres:
        url = str(row.o)
        if url.startswith('http') or url.startswith('https'):
            try:
                tld = url.split('/')[2].split('.')[-1]
                if 1 < len(tld) <= 10:
                    tlds.add(tld)
            except:
                pass
    return tlds

def select_local_property(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
    SELECT DISTINCT ?property
    WHERE {
        ?subject ?property ?object .
        FILTER isIRI(?property)
    }""")
    properties = []
    for row in qres:
        properties.append(str(row.property))
    return properties


def select_local_property_names(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
        SELECT DISTINCT ?property WHERE {
            ?subject ?property ?object .
            FILTER isIRI(?property) # Ensure it's a property URI
        }
    """,
    initNs={"rdf": rdflib.RDF}
    )

    local_property_names = []
    processed_local_names = set()

    for row in qres:
        property_uri = str(row.property)
        if not property_uri:
            continue

        local_name = ""
        if "#" in property_uri:
            local_name = property_uri.split("#")[-1]
        elif "/" in property_uri:
            local_name = property_uri.split("/")[-1]
        else:
            local_name = property_uri

        if local_name and local_name not in processed_local_names:
            local_property_names.append(local_name)
            processed_local_names.add(local_name)

    return local_property_names


def select_local_class_name(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
        SELECT DISTINCT ?classUri WHERE {
                ?s rdf:type ?classUri .
        }
        """,
        initNs={"rdf": rdflib.RDF}
    )
    local_names = []
    for row in qres:
        class_uri = str(row.classUri)
        if not class_uri:
            continue

        local_name = ""
        if "#" in class_uri:
            local_name = class_uri.split("#")[-1]
        elif "/" in class_uri:
            local_name = class_uri.split("/")[-1]
        else:
            local_name = class_uri
        local_names.append(local_name)
    return local_names

def create_local_dataset():
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    df = pd.DataFrame(columns=['id', 'voc', 'curi', 'puri', 'lcn', 'lpn', 'lab', 'tld', 'category'])
    for category in categories:
        for file in listdir(f'../data/raw/rdf_dump/{category}'):
            path = f'../data/raw/rdf_dump/{category}/{file}'
            number = re.search(r'(\d+)\.rdf', path)
            try:
                df.loc[len(df)] = [lod_frame['id'][int(number.group(1))],
                       select_local_vocabularies(path),
                       select_local_class(path),
                       select_local_property(path),
                       select_local_class_name(path),
                       select_local_property_names(path),
                       select_local_label(path),
                       select_local_tld(path),
                       lod_frame['category'][int(number.group(1))]]
            except Exception as e:
                print(e)
    df.to_csv('../data/raw/local_feature_set.csv', index=False)

MAX_OFFSET = 1000 # Define max offset to limit query size

def select_remote_vocabularies_sparqlwrapper(endpoint, limit=100, timeout=300, max_offset=MAX_OFFSET):
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout) # Set timeout
    vocabularies = set()
    offset = 0
    while True:
        if offset >= max_offset: # Check max offset
            break
        sparql.setQuery(f"""
            SELECT ?predicate
            WHERE {{
                ?subject ?predicate ?object .
            }}
            LIMIT {limit}
            OFFSET {offset}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query()
        try:
            results = results.convert()
        except Exception as e:
            print(f"Query execution error: {e}")
            break # Exit loop on error

        bindings = results["results"]["bindings"]
        if not bindings: # no more results
            break
        for result in bindings:
            predicate_uri = result["predicate"]["value"]
            if predicate_uri:
                vocabulary_uri = ""
                if "#" in predicate_uri:
                    vocabulary_uri = predicate_uri.split("#")[0]
                elif "/" in predicate_uri:
                    vocabulary_uri = predicate_uri.split("/")
                    vocabulary_uri = "/".join(vocabulary_uri[:len(vocabulary_uri)-1]) if len(vocabulary_uri) > 1 else predicate_uri # Get path before last part
                else:
                    vocabulary_uri = predicate_uri

                if vocabulary_uri and not vocabulary_uri.startswith("http://www.w3.org/"):
                    vocabularies.add(vocabulary_uri)
        offset += limit
    return vocabularies

def select_remote_class_sparqlwrapper(endpoint, limit=100, timeout=300, max_offset=MAX_OFFSET):
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout) # Set timeout
    classes = []
    offset = 0
    while True:
        if offset >= max_offset: # Check max offset
            break
        sparql.setQuery(f"""SELECT DISTINCT ?classUri
        WHERE {{
                ?classUri a ?type .
                FILTER (?type IN (rdfs:Class, owl:Class))
        }}
        LIMIT {limit}
        OFFSET {offset}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query()
        try:
            results = results.convert()
        except Exception as e:
            print(f"Query execution error: {e}")
            break # Exit loop on error
        bindings = results["results"]["bindings"]
        if not bindings:
            break
        for result in bindings:
            classes.append(result["classUri"]["value"])
        offset += limit
    return classes

def select_remote_label_sparqlwrapper(endpoint, limit=100, timeout=300, max_offset=MAX_OFFSET):
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout) # Set timeout
    labels = []
    offset = 0
    while True:
        if offset >= max_offset: # Check max offset
            break
        sparql.setQuery(f"""SELECT DISTINCT ?type
        WHERE {{
                ?class rdfs:label ?type .
        }}
        LIMIT {limit}
        OFFSET {offset}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query()
        try:
            results = results.convert()
        except Exception as e:
            print(f"Query execution error: {e}")
            break # Exit loop on error
        bindings = results["results"]["bindings"]
        if not bindings:
            break
        for result in bindings:
            labels.append(result["type"]["value"])
        offset += limit
    return labels


def select_remote_tld_sparqlwrapper(endpoint, limit=100, timeout=300, max_offset=MAX_OFFSET):
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout) # Set timeout
    tlds = set()
    offset = 0
    while True:
        if offset >= max_offset: # Check max offset
            break
        sparql.setQuery(f"""
        SELECT DISTINCT ?o
        WHERE {{
             ?s ?p ?o .
            FILTER(isIRI(?o))
        }}
        LIMIT {limit}
        OFFSET {offset}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query()
        try:
            results = results.convert()
        except Exception as e:
            print(f"Query execution error: {e}")
            break # Exit loop on error
        bindings = results["results"]["bindings"]
        if not bindings:
            break
        for result in bindings:
            url = result["o"]["value"]
            if url.startswith('http') or url.startswith('https'):
                try:
                    tld = url.split('/')[2].split('.')[-1]
                    if 1 < len(tld) <= 10:
                        tlds.add(tld)
                except:
                    pass
        offset += limit
    return tlds

def select_remote_property_sparqlwrapper(endpoint, limit=10, timeout=300, max_offset=MAX_OFFSET): # Reduced limit for potentially heavier query
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout) # Set timeout
    properties = []
    offset = 0
    while True:
        if offset >= max_offset: # Check max offset
            break
        sparql.setQuery(f"""
        SELECT DISTINCT ?property
        WHERE {{
            ?subject ?property ?object .
            FILTER isIRI(?property)
        }}
        LIMIT {limit}
        OFFSET {offset}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query()
        try:
            results = results.convert()
        except Exception as e:
            print(f"Query execution error: {e}")
            break # Exit loop on error
        bindings = results["results"]["bindings"]
        if not bindings:
            break
        for result in bindings:
            properties.append(result["property"]["value"])
        offset += limit
    return properties


def select_remote_property_names_sparqlwrapper(endpoint, limit=10, timeout=300, max_offset=MAX_OFFSET): # Reduced limit for potentially heavier query
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout) # Set timeout
    local_property_names = []
    processed_local_names = set()
    offset = 0
    while True:
        if offset >= max_offset: # Check max offset
            break
        sparql.setQuery(f"""
            SELECT DISTINCT ?property WHERE {{
                ?subject ?property ?object .
                FILTER isIRI(?property) # Ensure it's a property URI
            }}
            LIMIT {limit}
            OFFSET {offset}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query()
        try:
            results = results.convert()
        except Exception as e:
            print(f"Query execution error: {e}")
            break # Exit loop on error
        bindings = results["results"]["bindings"]
        if not bindings:
            break
        for result in bindings:
            property_uri = result["property"]["value"]
            if not property_uri:
                continue

            local_name = ""
            if "#" in property_uri:
                local_name = property_uri.split("#")[-1]
            elif "/" in property_uri:
                local_name = property_uri.split("/")[-1]
            else:
                local_name = property_uri

            if local_name and local_name not in processed_local_names:
                local_property_names.append(local_name)
                processed_local_names.add(local_name)
        offset += limit
    return local_property_names


def select_remote_class_name_sparqlwrapper(endpoint, limit=10, timeout=300, max_offset=MAX_OFFSET): # Reduced limit for potentially heavier query
    sparql = SPARQLWrapper(endpoint)
    sparql.setTimeout(timeout) # Set timeout
    local_names = []
    offset = 0
    while True:
        if offset >= max_offset: # Check max offset
            break
        sparql.setQuery(f"""
            SELECT DISTINCT ?classUri WHERE {{
                    ?s rdf:type ?classUri .
            }}
            LIMIT {limit}
            OFFSET {offset}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query()
        try:
            results = results.convert()
        except Exception as e:
            print(f"Query execution error: {e}")
            break # Exit loop on error
        bindings = results["results"]["bindings"]
        if not bindings:
            break
        for result in bindings:
            class_uri = result["classUri"]["value"]
            if not class_uri:
                continue

            local_name = ""
            if "#" in class_uri:
                local_name = class_uri.split("#")[-1]
            elif "/" in class_uri:
                local_name = class_uri.split("/")[-1]
            else:
                local_name = class_uri
            local_names.append(local_name)
        offset += limit
    return local_names

def create_remote_dataset_sparqlwrapper():
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    df = pd.DataFrame(columns=['id', 'voc', 'curi', 'puri', 'lcn', 'lpn', 'lab', 'tld', 'category'])
    for category in categories:
        for index, row in lod_frame.iterrows():
            if row['category'] == category and row['sparql_url']:
                logger.info(f"Endpoint: {row['id']} Number: {index}")
                endpoint = row['sparql_url']
                try:
                    df.loc[len(df)] = [row['id'],
                           select_remote_vocabularies_sparqlwrapper(endpoint),
                           select_remote_class_sparqlwrapper(endpoint),
                           select_remote_property_sparqlwrapper(endpoint),
                           select_remote_class_name_sparqlwrapper(endpoint),
                           select_remote_property_names_sparqlwrapper(endpoint),
                           select_remote_label_sparqlwrapper(endpoint),
                           select_remote_tld_sparqlwrapper(endpoint),
                           row['category']]
                    print(df[len(df) - 1])
                except Exception as e:
                    print(e)
    df.to_csv('../data/raw/remote_feature_set_sparqlwrapper.csv', index=False)

create_remote_dataset_sparqlwrapper()