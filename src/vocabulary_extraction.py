import json
import time

import pandas as pd
import rdflib.util
import requests
from SPARQLWrapper import SPARQLWrapper
from rdflib import Graph

from src.preprocessing import merge_dataset
from src.util import is_endpoint_working

return_formats = ['JSON', 'XML', 'TURTLE', 'N3', 'RDF', 'RDFXML', 'CSV', 'TSV', 'JSONLD']


def load_endpoint_list():
    return pd.read_csv("../data/raw/sparql_full_download.csv")['sparql_url']




def find_vocabulary_tag(endpoint, limit=5):
    query_string = """
   SELECT ?vocab
WHERE {
  ?s ?p ?vocab . # Bind ?vocab to the object of the triple
  FILTER(isIRI(?vocab)) # Ensure it's an IRI (vocabulary term)
} LIMIT 10"""
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query_string)
    sparql.setTimeout(120)
    try:
        sparql.setReturnFormat("json")
        results = sparql.query().convert()
        return _get_lov_tag(results)
    except Exception as e:
        return None


def find_vocabulary_local(path: str):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
   SELECT DISTINCT ?vocab
WHERE {
  ?s ?p ?vocab . # Bind ?vocab to the object of the triple
  FILTER(isIRI(?vocab)) # Ensure it's an IRI (vocabulary term)
  FILTER(!CONTAINS(STR(?vocab), "w3.org")) # Exclude results from w3.org
}""")
    for row in qres:
        print(row.vocab)
        print(requests.get(f"https://lov.linkeddata.es/dataset/lov/api/v2/vocabulary/info?vocab={row.vocab}").text)
        print(requests.get(f"https://coli-conc.gbv.de/api/concepts?uri={row.vocab}").text)


def _get_lov_tag(sparql_res):
    tags = []
    for result in sparql_res['results']['bindings']:
        res = result['vocab']['value']
        status = requests.get(f"https://lov.linkeddata.es/dataset/lov/api/v2/vocabulary/info?vocab={res}")
        if status.status_code != 404:
            text = json.loads(status.text)
            tags.append(text['tags'])
            print(text['tags'])
        else:
            status = requests.get(f"https://coli-conc.gbv.de/api/concepts?uri={res}")
            if status.status_code != 404:
                print(status.text)
        return tags


def search_from_list():
    for endpoint in load_endpoint_list():
        if is_endpoint_working(endpoint):
            find_vocabulary_tag(endpoint)


def find_tags_from_json():
    df = merge_dataset()
    response_df = pd.DataFrame(columns=['id', 'tags', 'category'])
    subject_list = []
    response_cache = {}
    for index, row in df.iterrows():
        for voc in set(row['voc']):
            print(f'Vocabulary: {voc}')
            if voc not in response_cache:
                response_lov = requests.get(f"https://lov.linkeddata.es/dataset/lov/api/v2/vocabulary/info?vocab={voc}",
                                            timeout=300)
                response_cache[voc] = response_lov
                time.sleep(0.1)
            else:
                response_lov = response_cache[voc]
            if response_lov.status_code != 404:
                try:
                    response_dict = json.loads(response_lov.text)
                    tags = response_dict['tags']
                    frame_tags = []
                    for tag in tags:
                        if 'Vocabularies' not in tag and 'Metadata' not in tag and 'FRBR' not in tag:
                            subject_list.append(tag)
                            frame_tags.append(tag)
                    response_df.loc[len(response_df)] = {
                        'id': row['id'],
                        'tags': frame_tags,
                        'category': row['category']
                    }

                    response_df.to_json('../data/raw/vocabulary_tag.json', index=False)
                except Exception as e:
                    print(e)
            print(requests.get(f"https://coli-conc.gbv.de/api/concepts?uri={voc}").text)

    return subject_list
