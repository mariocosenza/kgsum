import json
import rdflib.util
import requests
from SPARQLWrapper import SPARQLWrapper
from mpmath.libmp.backend import python3
from rdflib import Graph
import pandas as pd

return_formats = ['JSON', 'XML', 'TURTLE', 'N3', 'RDF', 'RDFXML', 'CSV', 'TSV', 'JSONLD']

def load_endpoint_list():
   return pd.read_csv("../data/raw/sparql_full_download.csv")['sparql_url']

def is_endpoint_working(endpoint)-> bool:
   query_string = """
      SELECT ?s ?p ?o
   WHERE {
      ?s ?p ?o
   } LIMIT 1"""
   sparql = SPARQLWrapper(endpoint)
   sparql.setQuery(query_string)
   sparql.setTimeout(120)
   try:
      result = sparql.query()
      str_header = result.response.headers.as_string()
      if 'text/plain' in str_header or 'text/html' in str_header or 'application/octet-stream' in str_header:
         return False
      result.convert()
      return True
   except Exception as e:
      return False


def find_vocabulary_tag(endpoint, limit = 5):
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
   df = pd.read_json("../data/raw/local_feature_set.json")
   response_df = pd.DataFrame(columns=['id', 'tags', 'category'])
   subject_list = []

   for index, row in df.iterrows():
      for voc in set(row['voc']):
         response_lov = requests.get(f"https://lov.linkeddata.es/dataset/lov/api/v2/vocabulary/info?vocab={voc}",
                                     timeout=120)
         if response_lov.status_code != 404:
            try:
               response_dict = json.loads(response_lov.text)
               tags = response_dict['tags']
               frame_tags = []
               for tag in tags:
                  if 'Vocabularies' not in tag and 'Metadata' not in tag:
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
