import requests
from SPARQLWrapper import SPARQLWrapper


def find_vocabulary_tag(endpoint, limit = 5):
   query_string = """
   SELECT DISTINCT ?vocab
WHERE {
  ?s ?p ?vocab . # Bind ?vocab to the object of the triple
  FILTER(isIRI(?vocab)) # Ensure it's an IRI (vocabulary term)
  FILTER(STRSTARTS(STR(?vocab), "http://") || STRSTARTS(STR(?vocab), "https://")) # Filter for common URI schemes
  FILTER(!CONTAINS(STR(?vocab), "w3.org")) # Exclude results from w3.org
  FILTER(!CONTAINS(STR(?vocab), "openlinksw.com")) # Exclude results from w3.org
} LIMIT 6"""
   sparql = SPARQLWrapper(endpoint)
   sparql.setQuery(query_string)
   try:
      sparql.setReturnFormat("json")
      results = sparql.query().convert()
      return get_lov_tag(results)
   except:
       print("Error")

def get_lov_tag(sparql_res):
   tags = []
   for result in sparql_res['results']['bindings']:
      res = result['vocab']['value']
      status = requests.get(f"https://lov.linkeddata.es/dataset/lov/api/v2/vocabulary/info?vocab={res}")
      if status.status_code != 404:
         tags.append(status.text)
      return tags
