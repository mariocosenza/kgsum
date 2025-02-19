from SPARQLWrapper import SPARQLWrapper


def is_endpoint_working(endpoint) -> bool:
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
