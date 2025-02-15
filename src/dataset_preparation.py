import rdflib
from rdflib import Graph

def select_local_vocabularies(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
        SELECT DISTINCT ?predicate
        WHERE {
            ?subject ?predicate ?object .
            FILTER (STRSTARTS(STR(?predicate), "http://"))
            FILTER (!STRSTARTS(STR(STRBEFORE(STR(?predicate), "#")), "http://www.w3.org/"))
        }
    """)
    vocabularies = []
    for row in qres:
        predicate_uri = str(row.predicate)
        if predicate_uri:
            vocabulary_uri = ""
            if "#" in predicate_uri:
                vocabulary_uri = predicate_uri.split("#")[0]
            elif "/" in predicate_uri:
                vocabulary_uri = predicate_uri.split("/")
                vocabulary_uri = "/".join(vocabulary_uri[:len(vocabulary_uri)-1]) if len(vocabulary_uri) > 1 else predicate_uri # Get path before last part
            else:
                vocabulary_uri = predicate_uri

            if vocabulary_uri and not vocabulary_uri.startswith("http://www.w3.org/"):
                vocabularies.append(vocabulary_uri)
    return vocabularies

def select_local_class(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""SELECT DISTINCT ?classUri
    WHERE {
            ?classUri a ?type .
            FILTER (?type IN (rdfs:Class, owl:Class))
    }""")
    classes = []
    for row in qres:
        classes.append(str(row.classUri))
    return classes

def select_local_label(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""SELECT DISTINCT ?type
    WHERE {
            ?class rdfs:label ?type .
    }""", initNs={"rdf": rdflib.RDF})
    labels = []
    for row in qres:
        labels.append(str(row.type))
    return labels


def select_local_tld(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
    SELECT DISTINCT ?o
    WHERE {
         ?s ?p ?o .
        FILTER(isIRI(?o))
    }
    """)
    tlds = []
    for row in qres:
        url = str(row.o)
        if url.startswith('http') or url.startswith('https'):
            try:
                tld = url.split('/')[2].split('.')[-1]
                if 1 < len(tld) <= 10:
                    tlds.append(tld)
            except:
                pass
    return tlds

def select_local_property(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
    SELECT DISTINCT ?property
    WHERE {
        ?subject ?property ?object .
        FILTER isIRI(?property)
    }""")
    properties = []
    for row in qres:
        properties.append(str(row.property))
    return properties


def select_local_property_names(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
        SELECT DISTINCT ?property WHERE {
            ?subject ?property ?object .
            FILTER isIRI(?property) # Ensure it's a property URI
        }
    """,
    initNs={"rdf": rdflib.RDF}
    )

    local_property_names = []
    processed_local_names = set()

    for row in qres:
        property_uri = str(row.property)
        if not property_uri:
            continue

        local_name = ""
        if "#" in property_uri:
            local_name = property_uri.split("#")[-1]
        elif "/" in property_uri:
            local_name = property_uri.split("/")[-1]
        else:
            local_name = property_uri

        if local_name and local_name not in processed_local_names:
            local_property_names.append(local_name)
            processed_local_names.add(local_name)

    return local_property_names


def select_local_class_name(path):
    g = Graph()
    result = g.parse(path, format=rdflib.util.guess_format(path))
    qres = result.query("""
        SELECT DISTINCT ?classUri WHERE {
                ?s rdf:type ?classUri .
        }
        """,
        initNs={"rdf": rdflib.RDF}
    )
    local_names = []
    for row in qres:
        class_uri = str(row.classUri)
        if not class_uri:
            continue

        local_name = ""
        if "#" in class_uri:
            local_name = class_uri.split("#")[-1]
        elif "/" in class_uri:
            local_name = class_uri.split("/")[-1]
        else:
            local_name = class_uri
        local_names.append(local_name)
    return local_names
