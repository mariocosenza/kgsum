import logging
from multiprocessing import Pool
from os import listdir

import pandas as pd
import rdflib
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

from src.util import match_file_lod, CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset_preparation")

FORMATS = {'ttl', 'xml', 'nt', 'trig', 'n3', 'nquads'}

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
    SELECT ?classUri (COUNT(?instance) AS ?instanceCount)
    WHERE {
        ?instance a ?classUri .
    }
    GROUP BY ?classUri
    ORDER BY DESC(?instanceCount)
    LIMIT 1000
""")

Q_LOCAL_LABEL = prepareQuery("""
    SELECT DISTINCT ?o
    WHERE {
        {?class rdfs:label ?o}
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
""", initNs={"schema": 'http://schema.org',
             "skos": 'http://www.w3.org/2004/02/skos/core#',
             "rdfs": 'http://www.w3.org/2000/01/rdf-schema#',
             "foaf" : 'http://xmlns.com/foaf/0.1/',
             "awol": 'http://bblfish.net/work/atom-owl/2006-06-06/#',
             "wdrs": 'http://www.w3.org/2007/05/powder-s#',
             "skosxl": 'http://www.w3.org/2008/05/skos-xl#'})

Q_LOCAL_LABEL_EN = prepareQuery("""
    SELECT DISTINCT ?o
    WHERE {
        {?class rdfs:label ?o}
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
""", initNs={"schema": 'http://schema.org',
             "skos": 'http://www.w3.org/2004/02/skos/core#',
             "rdfs": 'http://www.w3.org/2000/01/rdf-schema#',
             "foaf" : 'http://xmlns.com/foaf/0.1/',
             "awol": 'http://bblfish.net/work/atom-owl/2006-06-06/#',
             "wdrs": 'http://www.w3.org/2007/05/powder-s#',
             "skosxl": 'http://www.w3.org/2008/05/skos-xl#'})

Q_LOCAL_TLD = prepareQuery("""
    SELECT DISTINCT ?o
    WHERE {
        ?s ?p ?o .
        FILTER(isIRI(?o))
    } LIMIT 1000
""")

Q_LOCAL_PROPERTY = prepareQuery("""
    SELECT ?property (COUNT(?s) AS ?usageCount)
    WHERE {{
        ?s ?property ?o .
        # Optional: Filter out rdf:type if you want to exclude it
        FILTER (?property != rdf:type)
    }}
    GROUP BY ?property
    ORDER BY DESC(?usageCount)
    LIMIT 1000
""", initNs={"rdf": rdflib.RDF})

Q_LOCAL_PROPERTY_NAMES = prepareQuery("""
    SELECT ?property (COUNT(?s) AS ?usageCount)
    WHERE {{
        ?s ?property ?o .
        # Optional: Filter out rdf:type if you want to exclude it
        FILTER (?property != rdf:type)
    }}
    GROUP BY ?property
    ORDER BY DESC(?usageCount)
    LIMIT 1000
""")

Q_LOCAL_CLASS_NAME = prepareQuery("""
    SELECT ?classUri (COUNT(?instance) AS ?instanceCount)
    WHERE {
        ?instance a ?classUri .
    }
    GROUP BY ?classUri
    ORDER BY DESC(?instanceCount)
    LIMIT 1000
""", initNs={"rdf": rdflib.RDF})

Q_LOCAL_VOID_DESCRIPTION = prepareQuery("""
    SELECT DISTINCT ?s
    WHERE {
        ?s rdf:type void:Dataset . #dcat
    } LIMIT 100
""", initNs={"rdf": rdflib.RDF, "void": 'http://rdfs.org/ns/void#'})

Q_LOCAL_DCTERMS_DESCRIPTION = prepareQuery(
    """
    SELECT ?desc WHERE {
            {?s dcterms:description ?desc}
            UNION
            {?s schema:description ?desc}
    } LIMIT 100
    """, initNs={"dcterms": 'http://purl.org/dc/terms/', "schema": 'http://schema.org/'})

Q_LOCAL_DCTERMS_TITLE = prepareQuery(
    """
    SELECT ?desc WHERE {
            ?s dcterms:title ?desc .
    } LIMIT 1
    """, initNs={"dcterms": 'http://purl.org/dc/terms/'})

Q_LOCAL_VOID_SPARQL = prepareQuery("""
    SELECT DISTINCT ?sparql
    WHERE {
        ?s void:sparqlEndpoint ?o.
    } LIMIT 2
""", initNs={"void": 'http://rdfs.org/ns/void#'})

Q_LOCAL_DCTERMS_CREATOR = prepareQuery(
    """
    SELECT ?desc WHERE {
            ?s dcterms:creator ?creator .
    } LIMIT 5
""", initNs={"dcterms": 'http://purl.org/dc/terms/'})

Q_LOCAL_DCTERMS_LICENSE = prepareQuery(
    """
    SELECT ?desc WHERE {
            ?s dcterms:license ?license . 
    } LIMIT 1
""", initNs={"dcterms": 'http://purl.org/dc/terms/'})


# Helper functions to run SPARQL queries on parsed RDF graphs
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
    return {str(row.o) for row in qres}


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
            except Exception as exc:
                logger.warning(f'Unable to find tld {exc}')
    return tlds


def select_local_property(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_PROPERTY)
    return {str(row.property) for row in qres}


def select_local_endpoint(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_VOID_SPARQL)
    return {str(row.sparql) for row in qres}


def select_local_creator(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_DCTERMS_CREATOR)
    return {str(row.creator) for row in qres}


def select_local_license(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_DCTERMS_LICENSE)
    return {str(row.license) for row in qres}


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


def select_local_void_title(parsed_graph):
    qres = parsed_graph.query(Q_LOCAL_DCTERMS_TITLE)
    return {row.desc for row in qres}


def _guess_format_and_parse(path):
    g = Graph()
    for f in FORMATS:
        try:
            return g.parse(path, format=f)
        except Exception as _:
            pass
    raise Exception('Format not supported')

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

        if not title or title == '':
            if sparql is not None:
                title = sparql


        return {
            'id': [title],
            'title': [title],
            'sbj': [subject],
            'dsc': [description],
            'voc': [vocabulary],
            'curi': [class_val],
            'puri': [property_val],
            'lcn': [cname],
            'lpn': [pname],
            'lab': [label],
            'sparql': [sparql],
            'tlds': [tld],
            'creator': [creator],
            'license': [licenses]
        }

    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {str(e)}")
        return None


# Processing functions for individual files
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
            select_local_endpoint(result),
            select_local_creator(result),
            select_local_license(result),
            lod_frame['category'][file_num]
        ]
        return row
    except Exception as e:
        logger.warning(f"Error processing file {path}: {str(e)}")
        return None


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
            select_local_void_title(result),
            select_local_void_subject(result),
            select_local_void_description(result),
            lod_frame['category'][num]
        ]
        return row
    except Exception as e:
        logger.warning(f"Error processing file {path}: {str(e)}")
        return None


# Top-level helper functions for multiprocessing (must be pickleable)
def process_task(args):
    cat, file, lod_frame, offset, limit = args
    return process_local_dataset_file(cat, file, lod_frame, offset, limit)


def process_task_void(args):
    cat, file, lod_frame, offset, limit = args
    return process_local_void_dataset_file(cat, file, lod_frame, offset, limit)


# Dataset creation functions using multiprocessing.Pool
def create_local_dataset(offset=0, limit=10000):
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    tasks = []
    for category in CATEGORIES:
        directory = f'../data/raw/rdf_dump/{category}'
        for file in listdir(directory):
            tasks.append((category, file, lod_frame, offset, limit))
    total_tasks = len(tasks)
    results = []
    # Limit the pool to 8 processes
    with Pool(processes=8, maxtasksperchild=10) as pool:
        for i, result in enumerate(pool.imap_unordered(process_task, tasks), 1):
            if result is not None:
                results.append(result)
            logger.info(f"Progress: {i}/{total_tasks} tasks completed.")
    df = pd.DataFrame(
        results,
        columns=['id', 'voc', 'curi', 'puri', 'lcn', 'lpn', 'lab', 'tld', 'sparql', 'creator', 'license', 'category']
    )
    df.to_json(f'../data/raw/local/local_feature_set{offset}-{limit}.json', index=False)


def create_local_void_dataset(offset=0, limit=10000):
    lod_frame = pd.read_csv('../data/raw/sparql_full_download.csv')
    tasks = []
    for category in CATEGORIES:
        directory = f'../data/raw/rdf_dump/{category}'
        for file in listdir(directory):
            tasks.append((category, file, lod_frame, offset, limit))
    total_tasks = len(tasks)
    results = []
    # Limit the pool to 8 processes here as well
    with Pool(processes=8, maxtasksperchild=10) as pool:
        for i, result in enumerate(pool.imap_unordered(process_task_void, tasks), 1):
            if result is not None:
                results.append(result)
            logger.info(f"(Void) Progress: {i}/{total_tasks} tasks completed.")
    df = pd.DataFrame(results, columns=['id', 'title', 'sbj', 'dsc', 'category'])
    df.to_json(f'../data/raw/local/local_void_feature_set{offset}-{limit}.json', index=False)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
   # create_local_void_dataset(offset=1101, limit=1125)  # Process items 1101 to 1150 (50 items)
    #time.sleep(120)
    #create_local_void_dataset(offset=1126, limit=1150)
   # create_local_void_dataset(offset=1151, limit=1200)  # Process items 1151 to 1200 (50 items)
    #create_local_void_dataset(offset=1201, limit=1250)
  #  create_local_void_dataset(offset=1251, limit=1300)  # Process items 1151 to 1200 (50 items)
   # create_local_void_dataset(offset=1301, limit=1350)
   # create_local_void_dataset(offset=1351, limit=1400)


    #create_local_dataset(offset=1201, limit=1250)  # Process items 1201 to 1250 (50 items)
    #time.sleep(120)
    #create_local_dataset(offset=1251, limit=1300)  # Process items 1251 to 1300 (50 items)







