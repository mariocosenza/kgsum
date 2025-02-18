import re
from os import listdir

import pandas as pd
import rdflib
from rdflib import Graph
from rdflib.term import _toPythonMapping

from service.endpoint_lod import logger


def force_string_converter(value):
    return str(value)
for dt in list(_toPythonMapping.keys()):
   _toPythonMapping[dt] = force_string_converter

categories = {
    'cross_domain', 'geography', 'government', 'life_sciences', 'linguistics', 'media', 'publications',
    'social_networking', 'user_generated'
}

formats = {
    'xml', 'turtle', 'json-ld', 'ntriples', 'n3', 'trig', 'trix', 'nquads'
}

def select_local_vocabularies(parsed_graph):
    qres = parsed_graph.query("""
        SELECT DISTINCT ?predicate
        WHERE {
            ?subject ?predicate ?object .
            FILTER (STRSTARTS(STR(?predicate), "http://"))
            FILTER (!STRSTARTS(STR(STRBEFORE(STR(?predicate), "#")), "http://www.w3.org/"))
        }
    """)
    vocabularies = set()
    for row in qres:
        predicate_uri = str(row.predicate)
        if predicate_uri:
            if "#" in predicate_uri:
                vocabulary_uri = predicate_uri.split("#")[0]
            elif "/" in predicate_uri:
                vocabulary_uri = predicate_uri.split("/")
                vocabulary_uri = "/".join(vocabulary_uri[:len(vocabulary_uri) - 1]) if len(
                    vocabulary_uri) > 1 else predicate_uri  # Get path before last part
            else:
                vocabulary_uri = predicate_uri

            if vocabulary_uri and not vocabulary_uri.startswith("http://www.w3.org/"):
                vocabularies.add(vocabulary_uri)
    return vocabularies


def select_local_class(parsed_graph):
    qres = parsed_graph.query("""SELECT DISTINCT ?classUri
    WHERE {
            ?classUri a ?type .
            FILTER (?type IN (rdfs:Class, owl:Class))
    }""")
    classes = []
    for row in qres:
        classes.append(str(row.classUri))
    return classes


def select_local_label(parsed_graph):
    qres = parsed_graph.query("""SELECT DISTINCT ?type
    WHERE {
            ?class rdfs:label ?type .
    }""", initNs={"rdf": rdflib.RDF})
    labels = set()
    for row in qres:
        labels.add(str(row.type))
    return labels


def select_local_tld(parsed_graph):
    qres = parsed_graph.query("""
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


def select_local_property(parsed_graph):
    qres = parsed_graph.query("""
    SELECT DISTINCT ?property
    WHERE {
        ?subject ?property ?object .
        FILTER isIRI(?property)
    }""")
    properties = set()
    for row in qres:
        properties.add(str(row.property))
    return properties


def select_local_property_names(parsed_graph):
    qres = parsed_graph.query("""
        SELECT DISTINCT ?property WHERE {
            ?subject ?property ?object .
            FILTER isIRI(?property) # Ensure it's a property URI
        }
    """,initNs={"rdf": rdflib.RDF})

    local_property_names = set()
    processed_local_names = set()

    for row in qres:
        property_uri = str(row.property)
        if not property_uri:
            continue

        if "#" in property_uri:
            local_name = property_uri.split("#")[-1]
        elif "/" in property_uri:
            local_name = property_uri.split("/")[-1]
        else:
            local_name = property_uri

        if local_name and local_name not in processed_local_names:
            local_property_names.add(local_name)
            processed_local_names.add(local_name)

    return local_property_names


def select_local_class_name(parsed_graph):
    qres = parsed_graph.query("""
        SELECT DISTINCT ?classUri WHERE {
                ?s rdf:type ?classUri .
        }
        """, initNs={"rdf": rdflib.RDF})
    local_names = set()
    for row in qres:
        class_uri = str(row.classUri)
        if not class_uri:
            continue

        if "#" in class_uri:
            local_name = class_uri.split("#")[-1]
        elif "/" in class_uri:
            local_name = class_uri.split("/")[-1]
        else:
            local_name = class_uri
        local_names.add(local_name)
    return local_names


def select_local_void_subject(parsed_graph):
    qres = parsed_graph.query("""
        SELECT ?classUri WHERE {
                ?s dcterms:subject ?classUri .
        } """, initNs={"dcterms": 'http://purl.org/dc/terms/'})
    local_names = set()
    for row in qres:
        local_names.add(row.classUri)

    return local_names

def select_local_void_description(parsed_graph):
    qres = parsed_graph.query("""
        SELECT ?classUri WHERE {
                ?s dcterms:description ?classUri .
        }""", initNs={"dcterms": 'http://purl.org/dc/terms/'}
                        )
    local_names = set()
    for row in qres:
        local_names.add(row.classUri)

    return local_names

def _guess_format_and_parse(path):
    g = Graph()
    for f in formats:
        try:
             return g.parse(path, format=f)
        except Exception as _:
            pass
    return g.parse(path, format='turtle')



def create_local_dataset():
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    df = pd.DataFrame(columns=['id', 'voc', 'curi', 'puri', 'lcn', 'lpn', 'lab', 'tld', 'category'])
    for category in categories:
        for file in listdir(f'../data/raw/rdf_dump/{category}'):
            path = f'../data/raw/rdf_dump/{category}/{file}'
            number = re.search(r'(\d+)\.rdf', path)
            try:
                logger.info(f'Processing graph id: {lod_frame['id'][int(number.group(1))]}')
                result = _guess_format_and_parse(path)
                df.loc[len(df)] = [lod_frame['id'][int(number.group(1))],
                                   select_local_vocabularies(result),
                                   select_local_class(result),
                                   select_local_property(result),
                                   select_local_class_name(result),
                                   select_local_property_names(result),
                                   select_local_label(result),
                                   select_local_tld(result),
                                   lod_frame['category'][int(number.group(1))]]
            except Exception as e:
                logger.warning(e)
    df.to_json('../data/raw/local_feature_set.json', index=False)


def create_local_void_dataset():
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    df = pd.DataFrame(columns=['id', 'sbj', 'dsc', 'category'])
    for category in categories:
        for file in listdir(f'../data/raw/rdf_dump/{category}'):
            path = f'../data/raw/rdf_dump/{category}/{file}'
            number = re.search(r'(\d+)\.rdf', path)
            try:
                result = _guess_format_and_parse(path)
                logger.info(f'Processing graph with void id: {lod_frame['id'][int(number.group(1))]}')
                df.loc[len(df)] = [lod_frame['id'][int(number.group(1))],
                                   select_local_void_subject(result),
                                   select_local_void_description(result),
                                   lod_frame['category'][int(number.group(1))]]
            except Exception as e:
                logger.warning(e)
    df.to_json('../data/raw/local_void_feature_set.json', index=False)

create_local_void_dataset()
