import xml.etree.ElementTree as ET
import lxml.etree as le
import pandas as pd

namespaces = {
    "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#": "fma:",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#": "nci:",
    "http://www.ihtsdo.org/snomed#": "snomed:",
}
na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})

def read_oaei_mappings(rdf_file, src_iri=None, tgt_iri=None):
    """
    Args:
        rdf_file: path to mappings in rdf format
        src_onto: source ontology URI
        tgt_onto: target ontology URI
        include_measure: including measure value or not

    Returns:
        mappings(=;>,<), mappings(?)
    """
    xml_root = ET.parse(rdf_file).getroot()
    ref_mappings = []  # where relation is "="
    ignored_mappings = []  # where relation is "?"

    for elem in xml_root.iter():
        # every Cell contains a mapping of en1 -rel(some value)-> en2
        if 'Cell' in elem.tag:
            for sub_elem in elem:
                if "entity1" in sub_elem.tag:
                    en1 = list(sub_elem.attrib.values())[0]
                elif "entity2" in sub_elem.tag:
                    en2 = list(sub_elem.attrib.values())[0]
                elif "relation" in sub_elem.tag:
                    rel = sub_elem.text
                elif "measure" in sub_elem.tag:
                    measure = sub_elem.text
            en1 = en1.replace(src_iri, namespaces[src_iri])
            en2 = en2.replace(tgt_iri, namespaces[tgt_iri])
            row = [en1, en2, measure]
            # =: equivalent; > superset of; < subset of.
            if rel == "=" or rel == ">" or rel == "<":
                # rel.replace("&gt;", ">").replace("&lt;", "<")
                ref_mappings.append(row)
            elif rel == "?":
                ignored_mappings.append(row)
            else:
                print("Unknown Relation Warning: ", rel)

    print("#Maps (\"=\"):", len(ref_mappings))
    print("#Maps (\"?\"):", len(ignored_mappings))

    return ref_mappings, ignored_mappings

def save_oaei_mappings(rdf_file, src_onto=None, tgt_onto=None):
    output_dir = rdf_file.replace(".rdf", "")
    ref_mappings, ignored_mappings = read_oaei_mappings(rdf_file, src_onto, tgt_onto)
    _df = pd.DataFrame(ref_mappings, columns=["Entity1", "Entity2", "Value"])
    _df.to_csv(output_dir + "_legal.tsv", index=False, sep='\t')
    _df = pd.DataFrame(ignored_mappings, columns=["Entity1", "Entity2", "Value"])
    _df.to_csv(output_dir + "_illegal.tsv", index=False, sep='\t')

def create_rdf_template(rdf_file):
    """Create rdf template without mappings"""
    root = le.parse(rdf_file).getroot()
    align = root[0]
    for elem in align:
        if "map" in elem.tag:
            align.remove(elem)
    with open("template.rdf", "wb+") as f:
        le.ElementTree(root).write(f, pretty_print=True, xml_declaration=True, encoding="utf-8")

def logmap_text_to_rdf(logmap_text, rdf_template):
    """Write LogMap output text into RDF format with given template
    
    Args:
        logmap_text: path to the LogMap output text file
        rdf_template: rdf template with no mappings
    """
    # read mappings
    mappings = []
    with open(logmap_text, "r") as f:
        for line in f.readlines():
            mapping = line.split("|")
            # only store the complete mappings
            if len(mapping) == 5:
                mappings.append(tuple(mapping[:4]))

    # create xml tree
    parser = le.XMLParser(remove_blank_text=True)
    root = le.parse(rdf_template, parser).getroot()
    align = root[0]
    for m in mappings:
        en1, en2, rel, measure = m
        rdf_map = le.SubElement(align, "map")
        rdf_cell = le.SubElement(rdf_map, "Cell")
        le.SubElement(rdf_cell, "entity1").set("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", en1)
        le.SubElement(rdf_cell, "entity2").set("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", en2)
        rdf_measure = le.SubElement(rdf_cell, "measure")
        rdf_measure.text = measure
        rdf_measure.set("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}datatype",
                        "http://www.w3.org/2001/XMLSchema#float")
        le.SubElement(rdf_cell, "relation").text = "="  # according to Ernesto, class subsumptions should be considered

    output_rdf = logmap_text.replace(".txt", ".rdf")
    with open(output_rdf, "wb+") as f:
        le.ElementTree(root).write(f, pretty_print=True, xml_declaration=True, encoding="utf-8")
