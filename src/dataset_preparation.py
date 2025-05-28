import logging
import os
from multiprocessing import get_context
from os import listdir

import pandas as pd
import rdflib
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

from src.util import match_file_lod, CATEGORIES
from util import is_voc_allowed, is_curi_allowed

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
            FILTER (STRSTARTS(STR(?predicate), "http://"))
            FILTER (!STRSTARTS(STR(STRBEFORE(STR(?predicate), "#")), "http://www.w3.org/"))
        } LIMIT 1000
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
        if predicate_uri:
            if "#" in predicate_uri:
                vocabulary_uri = predicate_uri.split("#")[0]
            elif "/" in predicate_uri:
                parts = predicate_uri.split("/")
                vocabulary_uri = "/".join(parts[:-1]) if len(parts) > 1 else predicate_uri
            else:
                vocabulary_uri = predicate_uri
            if vocabulary_uri and is_voc_allowed(vocabulary_uri):
                vocabularies.add(vocabulary_uri)
    return vocabularies


def select_local_class(parsed_graph):
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
        predicate_uri = str(row.classUri)
        if is_curi_allowed(predicate_uri):
            classes.add(predicate_uri)
    return list(classes)


def select_local_label(parsed_graph):
    Q_LOCAL_LABEL = prepareQuery("""
        SELECT DISTINCT ?o
        WHERE {
            ?s a ?label
            {?s rdfs:label ?o}
            UNION
            {?s foaf:name ?o}
            UNION
            {?s skos:prefLabel ?o}
            UNION
            {?s rdfs:comment ?o}
            UNION
            {?s awol:label ?o}
            UNION
            {?s skos:note ?o}
            UNION
            {?s wdrs:text ?o}
            UNION
            {?s skosxl:prefLabel ?o}
            UNION
            {?s skosxl:literalForm ?o}
            UNION
            {?s schema:name ?o}
        } LIMIT 1000
    """, initNs={
        "schema": 'http://schema.org',
        "skos": 'http://www.w3.org/2004/02/skos/core#',
        "rdfs": 'http://www.w3.org/2000/01/rdf-schema#',
        "foaf": 'http://xmlns.com/foaf/0.1/',
        "awol": 'http://bblfish.net/work/atom-owl/2006-06-06/#',
        "wdrs": 'http://www.w3.org/2007/05/powder-s#',
        "skosxl": 'http://www.w3.org/2008/05/skos-xl#'
    })
    Q_LOCAL_LABEL_EN = prepareQuery("""
        SELECT DISTINCT ?o
        WHERE {
            ?s a ?label
            {?s rdfs:label ?o}
            UNION
            {?s foaf:name ?o}
            UNION
            {?s skos:prefLabel ?o}
            UNION
            {?s rdfs:comment ?o}
            UNION
            {?s awol:label ?o}
            UNION
            {?s skos:note ?o}
            UNION
            {?s wdrs:text ?o}
            UNION
            {?s skosxl:prefLabel ?o}
            UNION
            {?s skosxl:literalForm ?o}
            UNION
            {?s schema:name ?o}
            FILTER(langMatches(lang(?o), "en"))
        } LIMIT 1000
    """, initNs={
        "schema": 'http://schema.org',
        "skos": 'http://www.w3.org/2004/02/skos/core#',
        "rdfs": 'http://www.w3.org/2000/01/rdf-schema#',
        "foaf": 'http://xmlns.com/foaf/0.1/',
        "awol": 'http://bblfish.net/work/atom-owl/2006-06-06/#',
        "wdrs": 'http://www.w3.org/2007/05/powder-s#',
        "skosxl": 'http://www.w3.org/2008/05/skos-xl#'
    })
    log_query(Q_LOCAL_LABEL_EN)
    try:
        qres = parsed_graph.query(Q_LOCAL_LABEL_EN)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_label (EN): {e}")
        qres = []
    if not qres or len(qres) < 2:
        log_query(Q_LOCAL_LABEL)
        try:
            qres = parsed_graph.query(Q_LOCAL_LABEL)
        except Exception as e:
            logger.warning(f"SPARQL error in select_local_label: {e}")
            return set()
    return {str(row.o) for row in qres}


def select_local_tld(parsed_graph):
    Q_LOCAL_TLD = prepareQuery("""
        SELECT DISTINCT ?o
        WHERE {
            ?s ?p ?o .
            FILTER(isIRI(?o))
        } LIMIT 1000
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
        if url.startswith('http') or url.startswith('https'):
            try:
                tld = url.split('/')[2].split('.')[-1]
                if 1 < len(tld) <= 10:
                    tlds.add(tld)
            except Exception as exc:
                logger.warning(f'Unable to find tld in {url}: {exc}')
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
        if is_voc_allowed(property_uri):
            properties.add(property_uri)
    return list(properties)


def select_local_endpoint(parsed_graph):
    Q_LOCAL_VOID_SPARQL = prepareQuery("""
        SELECT DISTINCT ?o
        WHERE {
            ?s void:sparqlEndpoint ?o.
        } LIMIT 2
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
        SELECT ?creator WHERE {
            ?s dcterms:creator ?creator .
        } LIMIT 5
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
        SELECT ?license WHERE {
            ?s dcterms:license ?license .
        } LIMIT 1
    """, initNs={"dcterms": 'http://purl.org/dc/terms/'})
    log_query(Q_LOCAL_DCTERMS_LICENSE)
    try:
        qres = parsed_graph.query(Q_LOCAL_DCTERMS_LICENSE)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_license: {e}")
        return set()
    return {str(row.license) for row in qres}


def select_local_property_names(parsed_graph):
    Q_LOCAL_PROPERTY_NAMES = prepareQuery("""
        SELECT ?property (COUNT(?s) AS ?usageCount)
        WHERE {
            ?s ?property ?o .
            FILTER (?property != rdf:type)
        }
        GROUP BY ?property
        ORDER BY DESC(?usageCount)
        LIMIT 1000
    """, initNs={"rdf": rdflib.RDF})
    log_query(Q_LOCAL_PROPERTY_NAMES)
    try:
        qres = parsed_graph.query(Q_LOCAL_PROPERTY_NAMES)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_property_names: {e}")
        return set()
    local_property_names = set()
    processed_local_names = set()
    for row in qres:
        property_uri = str(row.property)
        if not property_uri or not is_voc_allowed(property_uri):
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
    Q_LOCAL_CLASS_NAME = prepareQuery("""
        SELECT ?classUri (COUNT(?instance) AS ?instanceCount)
        WHERE {
            ?instance a ?classUri .
        }
        GROUP BY ?classUri
        ORDER BY DESC(?instanceCount)
        LIMIT 1000
    """, initNs={"rdf": rdflib.RDF})
    log_query(Q_LOCAL_CLASS_NAME)
    try:
        qres = parsed_graph.query(Q_LOCAL_CLASS_NAME)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_class_name: {e}")
        return set()
    local_names = set()
    for row in qres:
        class_uri = str(row.classUri)
        if not class_uri or not is_curi_allowed(class_uri):
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
    Q_LOCAL_VOID_DESCRIPTION = prepareQuery("""
        SELECT DISTINCT ?s
        WHERE {
            ?s rdf:type void:Dataset .
        } LIMIT 100
    """, initNs={"rdf": rdflib.RDF, "void": 'http://rdfs.org/ns/void#'})
    log_query(Q_LOCAL_VOID_DESCRIPTION)
    try:
        qres = parsed_graph.query(Q_LOCAL_VOID_DESCRIPTION)
    except Exception as e:
        logger.warning(f"SPARQL error in select_local_void_subject: {e}")
        return set()
    subject = set()
    for row in qres:
        query_str = f"""
            SELECT ?classUri
            WHERE {{
                <{row.s}> dcterms:subject ?classUri .
            }} LIMIT 100
        """
        log_query(query_str)
        try:
            result = parsed_graph.query(query_str, initNs={"dcterms": 'http://purl.org/dc/terms/'})
            for res in result:
                subject.add(str(res.classUri))
        except Exception as e:
            logger.warning(f"SPARQL error in select_local_void_subject loop: {e}")
    return subject


def select_local_void_description(parsed_graph):
    Q_LOCAL_DCTERMS_DESCRIPTION = prepareQuery("""
        SELECT ?desc WHERE {
            ?s dcterms:description ?desc
        } LIMIT 100
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
        SELECT ?desc WHERE {
            ?s dcterms:title ?desc .
        } LIMIT 1
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
            ?s owl:sameAs ?o
        } LIMIT 1000
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
    raise Exception(f'Format not supported for file: {path}')


def process_file_full_inplace(file_path) -> dict[str, list | set | str | None] | None:
    if file_path is None:
        return None
    try:
        logger.info(f"Processing graph id: {file_path}")
        result = _guess_format_and_parse(file_path)

        # Get all the values
        title = select_local_void_title(result)
        subject = select_local_void_subject(result)
        description = select_local_void_description(result)
        vocabulary = select_local_vocabularies(result)
        class_val = select_local_class(result)
        property_val = select_local_property(result)
        cname = select_local_class_name(result)
        pname = select_local_property_names(result)
        label = select_local_label(result)
        tld = select_local_tld(result)
        sparql = select_local_endpoint(result)
        creator = select_local_creator(result)
        licenses = select_local_license(result)
        con = select_local_con(result)

        if not title or title == '':
            if sparql is not None:
                title = sparql

        return {
            'id': title,
            'title': title,
            'sbj': list(subject),
            'dsc': list(description),
            'voc': list(vocabulary),
            'curi': list(class_val),
            'puri': list(property_val),
            'lcn': list(cname),
            'lpn': list(pname),
            'lab': list(label),
            'sparql': list(sparql),
            'tlds': list(tld),
            'creator': list(creator),
            'license': list(licenses),
            'con': con
        }

    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {str(e)}")
        return None


# --- Multiprocessing Setup ---
lod_frame_global = None


def init_worker(lod_frame_path):
    global lod_frame_global
    lod_frame_global = pd.read_csv(lod_frame_path)


def process_local_dataset_file(args):
    category, file, offset, limit = args
    global lod_frame_global
    path = os.path.join('../data/raw/rdf_dump', category, file)
    file_num = match_file_lod(file, limit, offset, lod_frame_global)
    if file_num is None:
        return None
    try:
        logger.info(f"Processing graph id: {lod_frame_global['id'][file_num]}")
        result = _guess_format_and_parse(path)
        row = [
            lod_frame_global['id'][file_num],
            list(select_local_vocabularies(result)),
            list(select_local_class(result)),
            list(select_local_property(result)),
            list(select_local_class_name(result)),
            list(select_local_property_names(result)),
            list(select_local_label(result)),
            list(select_local_tld(result)),
            list(select_local_endpoint(result)),
            list(select_local_creator(result)),
            list(select_local_license(result)),
            list(select_local_con(result)),
            lod_frame_global['category'][file_num]
        ]
        return row
    except Exception as e:
        logger.warning(f"Error processing file {path}: {str(e)}")
        return None


def process_local_void_dataset_file(args):
    category, file, offset, limit = args
    global lod_frame_global
    path = os.path.join('../data/raw/rdf_dump', category, file)
    num = match_file_lod(file, limit, offset, lod_frame_global)
    if num is None:
        return None
    try:
        result = _guess_format_and_parse(path)
        logger.info(f"Processing graph with void id: {lod_frame_global['id'][num]}")
        row = [
            lod_frame_global['id'][num],
            select_local_void_title(result),
            list(select_local_void_subject(result)),
            list(select_local_void_description(result)),
            lod_frame_global['category'][num]
        ]
        return row
    except Exception as e:
        logger.warning(f"Error processing file {path}: {str(e)}")
        return None


def robust_pool_map(pool, func, tasks, timeout=600):
    results = []
    # Use imap_unordered for true multiprocessing and immediate results
    it = pool.imap_unordered(func, tasks)
    for i, result in enumerate(it, 1):
        if result is not None:
            results.append(result)
        logger.info(f"Progress: {i}/{len(tasks)} tasks completed.")
    return results


def create_local_dataset(offset=0, limit=10000):
    lod_frame_path = '../data/raw/sparql_full_download.csv'
    tasks = []
    for category in CATEGORIES:
        directory = os.path.join('../data/raw/rdf_dump', category)
        if not os.path.isdir(directory):
            logger.warning(f"Directory not found: {directory}")
            continue
        for file in listdir(directory):
            if file.startswith('.'):
                continue
            tasks.append((category, file, offset, limit))
    if not tasks:
        logger.warning("No tasks scheduled for local dataset.")
        return
    ctx = get_context("spawn")
    with ctx.Pool(processes=8, maxtasksperchild=10, initializer=init_worker, initargs=(lod_frame_path,)) as pool:
        results = robust_pool_map(pool, process_local_dataset_file, tasks, timeout=6000)
    if results:
        df = pd.DataFrame(
            results,
            columns=['id', 'voc', 'curi', 'puri', 'lcn', 'lpn', 'lab', 'tlds', 'sparql', 'creator', 'license', 'con',
                     'category']
        )
        out_path = f'../data/raw/local/local_feature_set{offset}-{limit}.json'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_json(out_path, index=False)
    else:
        logger.warning("No results produced for local dataset.")


def create_local_void_dataset(offset=0, limit=10000):
    lod_frame_path = '../data/raw/sparql_full_download.csv'
    tasks = []
    for category in CATEGORIES:
        directory = os.path.join('../data/raw/rdf_dump', category)
        if not os.path.isdir(directory):
            logger.warning(f"Directory not found: {directory}")
            continue
        for file in listdir(directory):
            if file.startswith('.'):
                continue
            tasks.append((category, file, offset, limit))
    if not tasks:
        logger.warning("No tasks scheduled for local void dataset.")
        return
    ctx = get_context("spawn")
    with ctx.Pool(processes=8, maxtasksperchild=10, initializer=init_worker, initargs=(lod_frame_path,)) as pool:
        results = robust_pool_map(pool, process_local_void_dataset_file, tasks, timeout=600)
    if results:
        df = pd.DataFrame(results, columns=['id', 'title', 'sbj', 'dsc', 'category'])
        out_path = f'../data/raw/local/local_void_feature_set{offset}-{limit}.json'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_json(out_path, index=False)
    else:
        logger.warning("No results produced for local void dataset.")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()

    create_local_dataset(offset=0, limit=200)
    create_local_dataset(offset=1001, limit=1025)
    create_local_dataset(offset=1026, limit=1050)
    create_local_dataset(offset=1051, limit=1100)
    create_local_dataset(offset=1101, limit=1125)
    create_local_dataset(offset=1126, limit=1150)
    create_local_dataset(offset=1151, limit=1200)
    create_local_dataset(offset=1201, limit=1250)
    create_local_dataset(offset=1251, limit=1300)
    create_local_dataset(offset=1301, limit=1350)
    create_local_dataset(offset=2000, limit=4000)

    create_local_void_dataset(offset=0, limit=200)
    create_local_void_dataset(offset=1001, limit=1025)
    create_local_void_dataset(offset=1026, limit=1050)
    create_local_void_dataset(offset=1051, limit=1100)
    create_local_void_dataset(offset=1101, limit=1125)
    create_local_void_dataset(offset=1126, limit=1150)
    create_local_void_dataset(offset=1151, limit=1200)
    create_local_void_dataset(offset=1201, limit=1250)
    create_local_void_dataset(offset=1251, limit=1300)
    create_local_void_dataset(offset=1301, limit=1350)
    create_local_void_dataset(offset=2000, limit=4000)
