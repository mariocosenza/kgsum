import logging
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from os import listdir

import pandas as pd
import rdflib
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

from src.util import match_file_lod, CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset_preparation")

# Initialize the rdflib graph with Oxigraph store.
rdflib.Graph(store="Oxigraph")

FORMATS = {
    'ox-nt', 'ox-nq', 'ox-ttl', 'ox-trig', 'ox-xml'
}

# Precompile SPARQL queries
Q_LOCAL_VOCABULARIES = prepareQuery("""
    SELECT DISTINCT ?predicate
    WHERE {
        ?subject ?predicate ?object .
        FILTER (STRSTARTS(STR(?predicate), "http://"))
        FILTER (!STRSTARTS(STR(STRBEFORE(STR(?predicate), "#")), "http://www.w3.org/"))
    } LIMIT 1000
""")
Q_LOCAL_CLASS = prepareQuery("""
    SELECT DISTINCT ?classUri
    WHERE {
        ?classUri a ?type .
        FILTER (?type IN (rdfs:Class, owl:Class))
    } LIMIT 1000
""", initNs={'rdfs': 'http://www.w3.org/2000/01/rdf-schema#', 'owl': 'http://www.w3.org/2002/07/owl#'})
Q_LOCAL_LABEL = prepareQuery("""
    SELECT DISTINCT ?type
    WHERE {
        ?class rdfs:label ?type .
    } LIMIT 1000
""", initNs={"rdfs": 'http://www.w3.org/2000/01/rdf-schema#'})

Q_LOCAL_LABEL_EN = prepareQuery("""
    SELECT DISTINCT ?type
    WHERE {
         ?item rdfs:label ?type .
         FILTER(langMatches(lang(?type), "en"))
    } LIMIT 1000
""", initNs={"rdfs": 'http://www.w3.org/2000/01/rdf-schema#'})

Q_LOCAL_TLD = prepareQuery("""
    SELECT DISTINCT ?o
    WHERE {
        ?s ?p ?o .
        FILTER(isIRI(?o))
    } LIMIT 1000
""")
Q_LOCAL_PROPERTY = prepareQuery("""
    SELECT DISTINCT ?property
    WHERE {
        ?subject ?property ?object .
        FILTER isIRI(?property)
    } LIMIT 1000
""")
Q_LOCAL_PROPERTY_NAMES = prepareQuery("""
    SELECT DISTINCT ?property
    WHERE {
        ?subject ?property ?object .
        FILTER isIRI(?property)
    } LIMIT 1000
""")
Q_LOCAL_CLASS_NAME = prepareQuery("""
    SELECT DISTINCT ?classUri
    WHERE {
        ?s rdf:type ?classUri .
    } LIMIT 1000
""", initNs={"rdf": rdflib.RDF})

Q_LOCAL_VOID_DESCRIPTION = prepareQuery("""
    SELECT DISTINCT ?s
    WHERE {
        ?s rdf:type void:Dataset .
    } LIMIT 100
""", initNs={"rdf": rdflib.RDF, "void": 'http://rdfs.org/ns/void#'})

Q_LOCAL_DCTERMS_DESCRIPTION = prepareQuery(
    """
    SELECT ?desc WHERE {
            ?s dcterms:description ?desc .
    } LIMIT 1
    """, initNs={"dcterms": 'http://purl.org/dc/terms/'})


def select_local_vocabularies(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_VOCABULARIES)
    vocabularies = set()
    for row in qres:
        predicate_uri = str(row.predicate)
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
    return vocabularies


def select_local_class(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_CLASS)
    return [str(row.classUri) for row in qres]


def select_local_label(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_LABEL_EN)
    if len(qres) < 2:
        qres = parsed_graph.query(Q_LOCAL_LABEL)
    return {str(row.type) for row in qres}


def select_local_tld(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_TLD)
    tlds = set()
    for row in qres:
        url = str(row.o)
        if url.startswith('http') or url.startswith('https'):
            try:
                tld = url.split('/')[2].split('.')[-1]
                if 1 < len(tld) <= 10:
                    tlds.add(tld)
            except Exception:
                pass
    return tlds


def select_local_property(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_PROPERTY)
    return {str(row.property) for row in qres}


def select_local_property_names(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_PROPERTY_NAMES, initNs={"rdf": rdflib.RDF})
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
    qres = parsed_graph.query(Q_LOCAL_CLASS_NAME)
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
    qres = parsed_graph.query(Q_LOCAL_VOID_DESCRIPTION)
    subject = set()
    for row in qres:
        result = parsed_graph.query(f"""
            SELECT ?classUri
        WHERE {{
        <{row.s}> dcterms:subject ?classUri .
        }} LIMIT 100
        """, initNs={"dcterms": 'http://purl.org/dc/terms/'})
        for res in result:
            subject.add(res.classUri)
    return {row for row in subject}


def select_local_void_description(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_DCTERMS_DESCRIPTION)
    return {row.desc for row in qres}


def _guess_format_and_parse(path):
    g = Graph()
    for f in FORMATS:
        try:
            return g.parse(path, format=f)
        except Exception:
            pass
    raise Exception('Format not supported')


def process_local_dataset_file(category, file, lod_frame, offset, limit):
    path = f'../data/raw/rdf_dump/{category}/{file}'
    file_num = match_file_lod(file, limit, offset, lod_frame)
    if file_num is None:
        return None
    try:
        logger.info(f"Processing graph id: {lod_frame['id'][file_num]}")
        result = _guess_format_and_parse(path)
        row = [
            lod_frame['id'][file_num],
            select_local_vocabularies(result),
            select_local_class(result),
            select_local_property(result),
            select_local_class_name(result),
            select_local_property_names(result),
            select_local_label(result),
            select_local_tld(result),
            lod_frame['category'][file_num]
        ]
        return row
    except Exception as e:
        logger.warning(f"Error processing file {path}: {str(e)}")
        return None


def create_local_dataset(offset=0, limit=10000):
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    tasks = []
    for category in CATEGORIES:
        directory = f'../data/raw/rdf_dump/{category}'
        for file in listdir(directory):
            tasks.append((category, file))

    total_tasks = len(tasks)
    tasks_done = 0
    results = []
    n_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        task_iter = iter(tasks)
        future_to_task = {}
        # Submit initial tasks up to the number of workers
        for _ in range(n_workers):
            try:
                cat, file = next(task_iter)
                future = executor.submit(process_local_dataset_file, cat, file, lod_frame, offset, limit)
                future_to_task[future] = (cat, file)
            except StopIteration:
                break

        # Process futures as they complete and schedule new tasks
        while future_to_task:
            done, _ = wait(future_to_task.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    result = future.result()
                except Exception as e:
                    logger.warning(f"Task error: {e}")
                    result = None
                tasks_done += 1
                logger.info(f"Progress: {tasks_done}/{total_tasks} tasks completed.")
                if result is not None:
                    results.append(result)
                del future_to_task[future]
                try:
                    cat, file = next(task_iter)
                    new_future = executor.submit(process_local_dataset_file, cat, file, lod_frame, offset, limit)
                    future_to_task[new_future] = (cat, file)
                except StopIteration:
                    continue

    df = pd.DataFrame(results, columns=['id', 'voc', 'curi', 'puri', 'lcn', 'lpn', 'lab', 'tld', 'category'])
    df.to_json(f'../data/raw/local/local_feature_set{offset}-{limit}.json', index=False)


def process_local_void_dataset_file(category, file, lod_frame, offset, limit):
    path = f'../data/raw/rdf_dump/{category}/{file}'
    num = match_file_lod(file, limit, offset, lod_frame)
    if num is None:
        return None

    try:
        result = _guess_format_and_parse(path)
        logger.info(f"Processing graph with void id: {lod_frame['id'][num]}")
        row = [
            lod_frame['id'][num],
            select_local_void_subject(result),
            select_local_void_description(result),
            lod_frame['category'][num]
        ]
        return row
    except Exception as e:
        logger.warning(f"Error processing file {path}: {str(e)}")
        return None


def create_local_void_dataset(offset=0, limit=10000):
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    tasks = []
    for category in CATEGORIES:
        directory = f'../data/raw/rdf_dump/{category}'
        for file in listdir(directory):
            tasks.append((category, file))

    total_tasks = len(tasks)
    tasks_done = 0
    results = []
    n_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        task_iter = iter(tasks)
        future_to_task = {}
        for _ in range(n_workers):
            try:
                cat, file = next(task_iter)
                future = executor.submit(process_local_void_dataset_file, cat, file, lod_frame, offset, limit)
                future_to_task[future] = (cat, file)
            except StopIteration:
                break

        while future_to_task:
            done, _ = wait(future_to_task.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    result = future.result()
                except Exception as e:
                    logger.warning(f"(Void) Task error: {e}")
                    result = None
                tasks_done += 1
                logger.info(f"(Void) Progress: {tasks_done}/{total_tasks} tasks completed.")
                if result is not None:
                    results.append(result)
                del future_to_task[future]
                try:
                    cat, file = next(task_iter)
                    new_future = executor.submit(process_local_void_dataset_file, cat, file, lod_frame, offset, limit)
                    future_to_task[new_future] = (cat, file)
                except StopIteration:
                    continue

    df = pd.DataFrame(results, columns=['id', 'sbj', 'dsc', 'category'])
    df.to_json(f'../data/raw/local/local_void_feature_set{offset}-{limit}.json', index=False)


#if __name__ == '__main__':
    #import multiprocessing

    #multiprocessing.freeze_support()
    #create_local_dataset(offset=0, limit=200)
    # To process the void dataset, call:
    #create_local_void_dataset(offset=0, limit=200)
