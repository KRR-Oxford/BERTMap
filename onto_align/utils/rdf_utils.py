import xml.etree.ElementTree as ET
import lxml.etree as le
import pandas as pd
from onto_align.utils import onto_uri_dict, onto_uri_dict_inv


def read_onto_uris(rdf_file):
    xml_root = ET.parse(rdf_file).getroot()
    onto_uri1, onto_uri2 = "", ""
    for elem in xml_root.iter():
        if "uri1" in elem.tag:
            onto_uri1 = elem.text
        elif "uri2" in elem.tag:
            onto_uri2 = elem.text
        # end case when both ontologies are captured
        if not onto_uri1 == "" and not onto_uri2 == "":
            break
    return onto_uri1, onto_uri2


def reformat_entity_uri(entity_uri, onto_uri, inverse=False):
    """
    Returns: onto_uri#fragment <=> onto_prefix:fragment
    """
    if not inverse:
        entity_uri = entity_uri.replace(onto_uri + "#", onto_uri_dict[onto_uri] + ":")
    else:
        entity_uri = entity_uri.replace(onto_uri_dict_inv[onto_uri] + ":", onto_uri + "#")
    return entity_uri


def read_mappings(rdf_file, src_onto=None, tgt_onto=None, include_measure=False):
    """
    Args:
        rdf_file: path to mappings in rdf format
        src_onto: source ontology URI
        tgt_onto: target ontology URI
        include_measure: including measure value or not

    Returns:
        mappings(=), mappings(?)
    """
    xml_root = ET.parse(rdf_file).getroot()
    legal_mappings = []  # where relation is "="
    illegal_mappings = []  # where relation is "?"
    if src_onto is None or tgt_onto is None:
        # Read URIs for ontology 1 and 2 from rdf if not given
        src_onto, tgt_onto = read_onto_uris(rdf_file)

    for elem in xml_root.iter():
        # every Cell contains a mapping of en1 -rel(some value)-> en2
        if 'Cell' in elem.tag:
            # follow the order in the oaei rdf mappings
            en1, en2, measure, rel = tuple(list(elem))
            en1, en2 = list(en1.attrib.values())[0], list(en2.attrib.values())[0]
            en1, en2 = reformat_entity_uri(en1, src_onto), reformat_entity_uri(en2, tgt_onto)
            measure, rel = measure.text, rel.text
            row = [en1, en2, measure] if include_measure else [en1, en2]
            if rel == "=":
                legal_mappings.append(row)
            else:
                illegal_mappings.append(row)

    print("#Maps (\"=\"):", len(legal_mappings))
    print("#Maps (\"?\"):", len(illegal_mappings))

    return legal_mappings, illegal_mappings


def rdf_to_tsv(rdf_file):
    output_dir = rdf_file.replace(".rdf", "")
    legal_mappings, illegal_mappings = read_mappings(rdf_file)
    _df = pd.DataFrame(legal_mappings, columns=["Entity1", "Entity2", "Value"])
    _df.to_csv(output_dir + "_legal.tsv", index=False, sep='\t')
    _df = pd.DataFrame(illegal_mappings, columns=["Entity1", "Entity2", "Value"])
    _df.to_csv(output_dir + "_illegal.tsv", index=False, sep='\t')


def create_rdf_template(rdf_file):
    root = le.parse(rdf_file).getroot()
    align = root[0]
    for elem in align:
        if "map" in elem.tag:
            align.remove(elem)
    with open("template.rdf", "wb+") as f:
        le.ElementTree(root).write(f, pretty_print=True, xml_declaration=True, encoding="utf-8")


def logmap_text_to_rdf(logmap_text, rdf_template):

    """
    Write LogMap output text into RDF format with given template
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
        rdf_measure.set("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}datatype", "http://www.w3.org/2001/XMLSchema#float")
        le.SubElement(rdf_cell, "relation").text = rel

    output_rdf = logmap_text.replace(".txt", ".rdf")
    with open(output_rdf, "wb+") as f:
        le.ElementTree(root).write(f, pretty_print=True, xml_declaration=True, encoding="utf-8")


