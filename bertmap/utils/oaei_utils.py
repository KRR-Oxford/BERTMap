from re import sub
import xml.etree.ElementTree as ET
import lxml.etree as le
import pandas as pd
from bertmap.onto import OntoBox


na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})

def read_tsv_mappings(tsv_file, threshold=0.0):
    """read mappings from tsv file"""
    _df = pd.read_csv(tsv_file, sep="\t", na_values=na_vals, keep_default_na=False) if type(tsv_file) is str else tsv_file
    mappings = ["\t".join(_df.iloc[i][:-1]) for i in range(len(_df)) if _df.iloc[i][-1] >= threshold]
    return mappings


def read_rdf_mappings(rdf_file, src_onto=None, tgt_onto=None):
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
    legal_mappings = []  # where relation is "="
    illegal_mappings = []  # where relation is "?"
    if src_onto is None or tgt_onto is None:
        # Read URIs for ontology 1 and 2 from rdf if not given
        uris = OntoBox.read_onto_uris_from_rdf(rdf_file, "uri1", "uri2")
        src_onto, tgt_onto = uris["uri1"], uris["uri2"]

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
            en1, en2 = OntoBox.reformat_entity_uri(en1, src_onto), OntoBox.reformat_entity_uri(en2, tgt_onto)
            row = [en1, en2, measure]
            # =: equivalent; > superset of; < subset of.
            if rel == "=" or rel == ">" or rel == "<":
                # rel.replace("&gt;", ">").replace("&lt;", "<")
                legal_mappings.append(row)
            elif rel == "?":
                illegal_mappings.append(row)
            else:
                print("Unknown Relation Warning: ", rel)

    print("#Maps (\"=\"):", len(legal_mappings))
    print("#Maps (\"?\"):", len(illegal_mappings))

    return legal_mappings, illegal_mappings


def rdf_to_tsv_mappings(rdf_file, src_onto=None, tgt_onto=None):
    output_dir = rdf_file.replace(".rdf", "")
    legal_mappings, illegal_mappings = read_rdf_mappings(rdf_file, src_onto, tgt_onto)
    _df = pd.DataFrame(legal_mappings, columns=["Entity1", "Entity2", "Value"])
    _df.to_csv(output_dir + "_legal.tsv", index=False, sep='\t')
    _df = pd.DataFrame(illegal_mappings, columns=["Entity1", "Entity2", "Value"])
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
