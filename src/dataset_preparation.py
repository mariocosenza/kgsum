import logging
import os
from multiprocessing import get_context
from os import listdir
from typing import Any

import pandas as pd
import rdflib
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

from lov_data_preparation import find_tags_from_list, find_comments_from_lists
from src.util import match_file_lod, CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset_preparation")

FORMATS = {'ttl', 'xml', 'nt', 'trig', 'n3', 'nquads'}

def log_query(query):
    logger.info(f"SPARQL Query: {query}")

def select_local_vocabularies(parsed_graph):
    Q_LOCAL_VOCABULARIES = prepareQuery("""
        SELECT DISTINCT ?predicate
        WHERE {
            ?subject ?predicate ?object .
            FILTER(STRSTARTS(STR(?predicate), "http://"))
            FILTER(!STRSTARTS(STR(STRBEFORE(STR(?predicate), "#")), "http://www.w3.org/"))
        }
        LIMIT 1000
    """)
    log_query(Q_LOCAL_VOCABULARIES)
    try:
        qres = parsed_graph.query(Q_LOCAL_VOCABULARIES)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_vocabularies: {e}")
        return set()

    vocabularies = set()
    for row in qres:
        predicate_uri = str(row.predicate)
        if not predicate_uri:
            continue

        if "#" in predicate_uri:
            vocabulary_uri = predicate_uri.split("#")[0]
        elif "/" in predicate_uri:
            parts = predicate_uri.rstrip("/").split("/")
            vocabulary_uri = "/".join(parts[:-1]) if len(parts) > 1 else predicate_uri
        else:
            vocabulary_uri = predicate_uri

        if not vocabulary_uri:
            continue

        vocabularies.add(vocabulary_uri)

    return vocabularies

def select_local_class(parsed_graph) -> list[str]:
    Q_LOCAL_CLASS = prepareQuery("""
        SELECT ?classUri (COUNT(?instance) AS ?instanceCount)
        WHERE {
            ?instance a ?classUri .
        }
        GROUP BY ?classUri
        ORDER BY DESC(?instanceCount)
        LIMIT 1000
    """)
    log_query(Q_LOCAL_CLASS)
    try:
        qres = parsed_graph.query(Q_LOCAL_CLASS)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_class: {e}")
        return []

    classes = set()
    for row in qres:
        class_uri = str(row.classUri)
        if class_uri:
            classes.add(class_uri)
    return list(classes)

def select_local_label(parsed_graph):
    ns = {
        "schema": 'http://schema.org',
        "skos": 'http://www.w3.org/2004/02/skos/core#',
        "rdfs": 'http://www.w3.org/2000/01/rdf-schema#',
        "foaf": 'http://xmlns.com/foaf/0.1/',
        "awol": 'http://bblfish.net/work/atom-owl/2006-06-06/#',
        "wdrs": 'http://www.w3.org/2007/05/powder-s#',
        "skosxl": 'http://www.w3.org/2008/05/skos-xl#'
    }

    Q_LOCAL_LABEL_EN = prepareQuery("""
        SELECT DISTINCT ?o
        WHERE {
            ?s a ?label .
            { ?s rdfs:label ?o }
            UNION
            { ?s foaf:name ?o }
            UNION
            { ?s skos:prefLabel ?o }
            UNION
            { ?s rdfs:comment ?o }
            UNION
            { ?s awol:label ?o }
            UNION
            { ?s skos:note ?o }
            UNION
            { ?s wdrs:text ?o }
            UNION
            { ?s skosxl:prefLabel ?o }
            UNION
            { ?s skosxl:literalForm ?o }
            UNION
            { ?s schema:name ?o }
            FILTER(langMatches(lang(?o), "en"))
        }
        LIMIT 1000
    """, initNs=ns)
    log_query(Q_LOCAL_LABEL_EN)

    try:
        qres = parsed_graph.query(Q_LOCAL_LABEL_EN)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_label (EN): {e}")
        qres = []

    if not qres or len(qres) < 2:
        Q_LOCAL_LABEL = prepareQuery("""
            SELECT DISTINCT ?o
            WHERE {
                ?s a ?label .
                { ?s rdfs:label ?o }
                UNION
                { ?s foaf:name ?o }
                UNION
                { ?s skos:prefLabel ?o }
                UNION
                { ?s rdfs:comment ?o }
                UNION
                { ?s awol:label ?o }
                UNION
                { ?s skos:note ?o }
                UNION
                { ?s wdrs:text ?o }
                UNION
                { ?s skosxl:prefLabel ?o }
                UNION
                { ?s skosxl:literalForm ?o }
                UNION
                { ?s schema:name ?o }
            }
            LIMIT 1000
        """, initNs=ns)
        log_query(Q_LOCAL_LABEL)
        try:
            qres = parsed_graph.query(Q_LOCAL_LABEL)
        except Exception as e:
            logger.warning(f"SPARQL error in select_local_label (fallback): {e}")
            return set()

    return {str(row.o) for row in qres}

def select_local_tld(parsed_graph):
    Q_LOCAL_TLD = prepareQuery("""
        SELECT DISTINCT ?o
        WHERE {
            ?s ?p ?o .
            FILTER(isIRI(?o))
        }
        LIMIT 1000
    """)
    log_query(Q_LOCAL_TLD)
    try:
        qres = parsed_graph.query(Q_LOCAL_TLD)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_tld: {e}")
        return set()

    tlds = set()
    for row in qres:
        url = str(row.o)
        if url.startswith(("http://", "https://")):
            try:
                host = url.split("/")[2]
                tld = host.split(".")[-1]
                if 1 < len(tld) <= 10:
                    tlds.add(tld)
            except Exception as exc:
                logger.warning(f"Unable to parse TLD from {url}: {exc}")
    return tlds

def select_local_property(parsed_graph):
    Q_LOCAL_PROPERTY = prepareQuery("""
        SELECT ?property (COUNT(?s) AS ?usageCount)
        WHERE {
            ?s ?property ?o .
            FILTER (?property != rdf:type)
        }
        GROUP BY ?property
        ORDER BY DESC(?usageCount)
        LIMIT 1000
    """, initNs={"rdf": rdflib.RDF})
    log_query(Q_LOCAL_PROPERTY)
    try:
        qres = parsed_graph.query(Q_LOCAL_PROPERTY)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_property: {e}")
        return []

    properties = set()
    for row in qres:
        property_uri = str(row.property)
        if not property_uri:
            continue
        properties.add(property_uri)
    return list(properties)

def select_local_endpoint(parsed_graph):
    Q_LOCAL_VOID_SPARQL = prepareQuery("""
        SELECT DISTINCT ?o
        WHERE {
            ?s void:sparqlEndpoint ?o .
        }
        LIMIT 2
    """, initNs={"void": 'http://rdfs.org/ns/void#'})
    log_query(Q_LOCAL_VOID_SPARQL)
    try:
        qres = parsed_graph.query(Q_LOCAL_VOID_SPARQL)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_endpoint: {e}")
        return []
    return list({str(row.o) for row in qres})

def select_local_creator(parsed_graph):
    Q_LOCAL_DCTERMS_CREATOR = prepareQuery("""
        SELECT ?creator
        WHERE {
            ?s dcterms:creator ?creator .
        }
        LIMIT 5
    """, initNs={"dcterms": 'http://purl.org/dc/terms/'})
    log_query(Q_LOCAL_DCTERMS_CREATOR)
    try:
        qres = parsed_graph.query(Q_LOCAL_DCTERMS_CREATOR)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_creator: {e}")
        return set()
    return {str(row.creator) for row in qres}

def select_local_license(parsed_graph):
    Q_LOCAL_DCTERMS_LICENSE = prepareQuery("""
        SELECT ?license
        WHERE {
            ?s dcterms:license ?license .
        }
        LIMIT 1
    """, initNs={"dcterms": 'http://purl.org/dc/terms/'})
    log_query(Q_LOCAL_DCTERMS_LICENSE)
    try:
        qres = parsed_graph.query(Q_LOCAL_DCTERMS_LICENSE)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_license: {e}")
        return set()
    return {str(row.license) for row in qres}

def select_local_void_subject(parsed_graph):
    Q_LOCAL_VOID_SUBJECT = prepareQuery("""
        SELECT DISTINCT ?s
        WHERE {
            ?s rdf:type void:Dataset .
        }
        LIMIT 100
    """, initNs={"rdf": rdflib.RDF, "void": 'http://rdfs.org/ns/void#'})
    log_query(Q_LOCAL_VOID_SUBJECT)

    try:
        qres = parsed_graph.query(Q_LOCAL_VOID_SUBJECT)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_void_subject: {e}")
        return set()

    dataset_uris = {str(row.s) for row in qres if str(row.s)}

    subject = set()
    for ds_uri in dataset_uris:
        query_str = f"""
            SELECT ?classUri
            WHERE {{
                <{ds_uri}> dcterms:subject ?classUri .
            }}
            LIMIT 100
        """
        log_query(query_str)
        try:
            result = parsed_graph.query(query_str, initNs={"dcterms": 'http://purl.org/dc/terms/'})
            for res in result:
                class_uri = str(res.classUri)
                if class_uri:
                    subject.add(class_uri)
        except Exception as e:
            logger.warning(f"SPARQL error in select_local_void_subject loop: {e}")
    return subject

def select_local_void_description(parsed_graph):
    Q_LOCAL_DCTERMS_DESCRIPTION = prepareQuery("""
        SELECT ?desc
        WHERE {
            ?s dcterms:description ?desc .
        }
        LIMIT 100
    """, initNs={"dcterms": 'http://purl.org/dc/terms/'})
    log_query(Q_LOCAL_DCTERMS_DESCRIPTION)
    try:
        qres = parsed_graph.query(Q_LOCAL_DCTERMS_DESCRIPTION)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_void_description: {e}")
        return set()
    return {str(row.desc) for row in qres}

def select_local_void_title(parsed_graph):
    Q_LOCAL_DCTERMS_TITLE = prepareQuery("""
        SELECT ?desc
        WHERE {
            ?s dcterms:title ?desc .
        }
        LIMIT 1
    """, initNs={"dcterms": 'http://purl.org/dc/terms/'})
    log_query(Q_LOCAL_DCTERMS_TITLE)
    try:
        qres = parsed_graph.query(Q_LOCAL_DCTERMS_TITLE)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_void_title: {e}")
        return []
    return [str(row.desc) for row in qres]

def select_local_con(parsed_graph):
    Q_LOCAL_CON = prepareQuery("""
        SELECT DISTINCT ?o
        WHERE {
            ?s owl:sameAs ?o .
        }
        LIMIT 1000
    """, initNs={"owl": 'http://www.w3.org/2002/07/owl#'})
    log_query(Q_LOCAL_CON)
    try:
        qres = parsed_graph.query(Q_LOCAL_CON)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_con: {e}")
        return []
    return [str(row.o) for row in qres]

def _guess_format_and_parse(path):
    g = Graph()
    for f in FORMATS:
        try:
            return g.parse(path, format=f)
        except Exception:
            continue
    raise Exception(f"Format not supported for file: {path}")

def process_file_full_inplace(
    file_path: str,
    ingest_lov: bool = False
) -> dict[str, Any] | None:
    if not file_path:
        return None

    try:
        logger.info(f"Processing graph file: {file_path}")
        parsed_graph = _guess_format_and_parse(file_path)

        title_list = select_local_void_title(parsed_graph)
        void_subjects = select_local_void_subject(parsed_graph)
        void_descriptions = select_local_void_description(parsed_graph)
        vocabularies = select_local_vocabularies(parsed_graph)
        class_list = select_local_class(parsed_graph)
        property_list = select_local_property(parsed_graph)
        labels = select_local_label(parsed_graph)
        tlds = select_local_tld(parsed_graph)
        endpoints = select_local_endpoint(parsed_graph)
        creators = select_local_creator(parsed_graph)
        licenses = select_local_license(parsed_graph)
        connections = select_local_con(parsed_graph)

        title = title_list[0] if title_list else (endpoints[0] if endpoints else "")
        class_list = list(class_list)
        property_list = list(property_list)
        vocabularies = list(vocabularies)
        voc_tags = []
        comments = []
        if ingest_lov:
            voc_tags = find_tags_from_list(vocabularies)
            comments = find_comments_from_lists(curi_list=class_list, puri_list=property_list)

        return {
            "id": title,
            "title": title,
            "sbj": list(void_subjects),
            "dsc": list(void_descriptions),
            "voc": list(vocabularies),
            "curi": list(class_list),
            "puri": list(property_list),
            "lab": list(labels),
            "sparql": endpoints,
            "tlds": list(tlds),
            "creator": list(creators),
            "license": list(licenses),
            "con": connections,
            "tags": voc_tags,
            "comments": comments
        }

    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {e}")
        return None

lod_frame_global: pd.DataFrame = pd.DataFrame()

def init_worker(lod_frame_path: str):
    global lod_frame_global
    df = pd.read_csv(lod_frame_path)
    lod_frame_global = df[~df["category"].fillna("").str.strip().eq("user_generated")].reset_index(drop=True)

def process_local_dataset_file(args):
    category, filename, offset, limit = args
    global lod_frame_global

    path = os.path.join("../data/raw/rdf_dump", category, filename)
    file_num = match_file_lod(filename, limit, offset, lod_frame_global)
    if file_num is None:
        return None

    row_id = lod_frame_global.at[file_num, "id"]
    try:
        parsed_graph = _guess_format_and_parse(path)
        vocab = select_local_vocabularies(parsed_graph)
        classes = select_local_class(parsed_graph)
        props = select_local_property(parsed_graph)
        labels = select_local_label(parsed_graph)
        tlds = select_local_tld(parsed_graph)
        endpoints = select_local_endpoint(parsed_graph)
        creators = select_local_creator(parsed_graph)
        licenses = select_local_license(parsed_graph)
        connections = select_local_con(parsed_graph)

        return [
            row_id,
            list(vocab),
            list(classes),
            list(props),
            list(labels),
            list(tlds),
            endpoints,
            list(creators),
            list(licenses),
            connections,
            lod_frame_global.at[file_num, "category"]
        ]
    except Exception as e:
        logger.warning(f"Error processing file {path}: {e}")
        return None

def process_local_void_dataset_file(args):
    category, filename, offset, limit = args
    global lod_frame_global

    path = os.path.join("../data/raw/rdf_dump", category, filename)
    file_num = match_file_lod(filename, limit, offset, lod_frame_global)
    if file_num is None:
        return None

    try:
        parsed_graph = _guess_format_and_parse(path)
        title_list = select_local_void_title(parsed_graph)
        void_subjects = select_local_void_subject(parsed_graph)
        void_descriptions = select_local_void_description(parsed_graph)

        title = title_list[0] if title_list else ""

        return [
            lod_frame_global.at[file_num, "id"],
            title,
            list(void_subjects),
            list(void_descriptions),
            lod_frame_global.at[file_num, "category"],
        ]
    except Exception as e:
        logger.warning(f"Error processing file {path}: {e}")
        return None

def robust_pool_map(pool, func, tasks):
    results = []
    total = len(tasks)
    for i, result in enumerate(pool.imap_unordered(func, tasks), start=1):
        if result is not None:
            results.append(result)
        logger.info(f"Progress: {i}/{total} tasks completed.")
    return results

def create_local_dataset(
    offset: int = 0,
    limit: int = 10000,
):
    out_path = f"../data/raw/local/local_feature_set_{offset}_{limit}.json"
    # Check subito, uscita immediata se il file esiste
    if os.path.exists(out_path):
        logger.info(f"File already exists: {out_path} -- Skipping creation.")
        return

    lod_frame_path = "../data/raw/sparql_full_download.csv"
    tasks = []

    valid_categories = [cat for cat in CATEGORIES if cat != "user_generated"]

    for category in valid_categories:
        directory = os.path.join("../data/raw/rdf_dump", category)
        if not os.path.isdir(directory):
            logger.warning(f"Directory not found: {directory}")
            continue
        for filename in listdir(directory):
            if filename.startswith("."):
                continue
            tasks.append((category, filename, offset, limit))

    if not tasks:
        logger.warning("No tasks scheduled for local dataset.")
        return

    ctx = get_context("spawn")
    with ctx.Pool(
            processes=min(4, os.cpu_count() or 4),
            maxtasksperchild=4,
            initializer=init_worker,
            initargs=(lod_frame_path,),
    ) as pool:
        results = robust_pool_map(pool, process_local_dataset_file, tasks)

    if results:
        df = pd.DataFrame(
            results,
            columns=[
                "id",
                "voc",
                "curi",
                "puri",
                "lab",
                "tlds",
                "sparql",
                "creator",
                "license",
                "con",
                "category",
            ],
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_json(out_path, orient="records", index=False)
        logger.info(f"Saved local feature set to {out_path}")
    else:
        logger.warning("No results produced for local dataset.")


def create_local_void_dataset(offset: int = 0, limit: int = 10000):
    out_path = f"../data/raw/local/local_void_feature_set_{offset}_{limit}.json"
    # Check subito, uscita immediata se il file esiste
    if os.path.exists(out_path):
        logger.info(f"File already exists: {out_path} -- Skipping creation.")
        return

    lod_frame_path = "../data/raw/sparql_full_download.csv"
    tasks = []

    valid_categories = [cat for cat in CATEGORIES if cat != "user_generated"]

    for category in valid_categories:
        directory = os.path.join("../data/raw/rdf_dump", category)
        if not os.path.isdir(directory):
            logger.warning(f"Directory not found: {directory}")
            continue
        for filename in listdir(directory):
            if filename.startswith("."):
                continue
            tasks.append((category, filename, offset, limit))

    if not tasks:
        logger.warning("No tasks scheduled for local void dataset.")
        return

    ctx = get_context("spawn")
    with ctx.Pool(
            processes=min(4, os.cpu_count() or 4),
            maxtasksperchild=4,
            initializer=init_worker,
            initargs=(lod_frame_path,),
    ) as pool:
        results = robust_pool_map(pool, process_local_dataset_file, tasks)

    if results:
        df = pd.DataFrame(results, columns=[
            "id", "title", "sbj", "dsc", "category"
        ])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_json(out_path, orient="records", index=False)
        logger.info(f"Saved local void feature set to {out_path}")
    else:
        logger.warning("No results produced for local void dataset.")
