import rdflib
from rdflib import Graph


def select_local_class(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""SELECT DISTINCT ?class
    WHERE {
            ?class a ?type .
            FILTER (?type IN (rdfs:Class, owl:Class))
    }""")

def select_local_label(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""SELECT DISTINCT ?class
    WHERE {
            ?class rdfs:label ?type .
    }""")

def select_local_tld(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""SELECT DISTINCT ?tld
    WHERE {
        ?s ?p ?o .
        FILTER isIRI(?o)
        BIND (STR(?o) AS ?iri_str)
        BIND (REPLACE(?iri_str, "^https?://(?:www\\.)?([^/]+).*$", "$1") AS ?domain)
        BIND (strafter(?domain, ".") AS ?tld_candidate)

        FILTER (contains(?domain, ".")) # Ensure there's a dot to potentially separate domain and TLD
        FILTER (!STRSTARTS(?tld_candidate, ?domain)) # Avoid cases where strafter returns the whole domain
        FILTER (strlen(?tld_candidate) <= 10) # Basic TLD length filter (adjust as needed)
        FILTER (strlen(?tld_candidate) > 1)  # Basic TLD length filter (adjust as needed)
    }""")

def select_local_property(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
    SELECT DISTINCT ?property
    WHERE {
        ?subject ?property ?object .
        FILTER isIRI(?property)
    }""")

def select_local_property_label(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
    SELECT DISTINCT ?localName WHERE {
    { ?s ?p ?o } UNION { ?s a ?c } # Get both properties and classes
    BIND(IF(isURI(?p), STR(?p), IF(isURI(?c), STR(?c), "")) AS ?uri) # Get URI string
    FILTER(STRLEN(?uri) > 0) # Ensure we have a URI
    BIND(SUBSTR(
        ?uri,
        IF(CONTAINS(?uri, "#"),
           STRPOS(?uri, "#") + 1,
           IF(CONTAINS(?uri, "/"),
              STRPOS(REVERSE(?uri), "/") + 1,
              1))
       ) AS ?localName) # Extract local name
    }
    """)

def select_local_class_name(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
    SELECT DISTINCT ?localName WHERE {
    ?s a ?c .  # Get the classes
    BIND(STR(?c) AS ?uri) # Get URI string of the class
    FILTER(STRLEN(?uri) > 0) # Ensure we have a URI
    BIND(SUBSTR(
        ?uri,
        IF(CONTAINS(?uri, "#"),
           STRPOS(?uri, "#") + 1,
           IF(CONTAINS(?uri, "/"),
              STRPOS(REVERSE(?uri), "/") + 1,
              1))
       ) AS ?localName) # Extract local name
    } 
    """)