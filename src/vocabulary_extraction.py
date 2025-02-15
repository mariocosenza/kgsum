import json
import rdflib.util
import requests
from SPARQLWrapper import SPARQLWrapper
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




#find_vocabulary_local('../data/raw/rdf_dump/geography/20.rdf')