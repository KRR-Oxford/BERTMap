import xml.etree.ElementTree as ET
import pandas as pd
from onto_align.utils import onto_uri_dict, onto_uri_dict_inv
# from onto_align import main_dir


def rdf_to_xml(rdf_file):
    return ET.parse(rdf_file).getroot()


def read_onto_uris(xml_root):
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
    :return: onto_uri#fragment <=> onto_prefix:fragment
    """
    if not inverse:
        entity_uri = entity_uri.replace(onto_uri + "#", onto_uri_dict[onto_uri] + ":")
    else:
        entity_uri = entity_uri.replace(onto_uri_dict_inv[onto_uri] + ":", onto_uri + "#")
    return entity_uri


def read_mappings(xml_root):
    legal_mappings = []  # where relation is "="
    illegal_mappings = []  # where relation is "?"
    onto1, onto2 = read_onto_uris(xml_root)  # URIs for ontology 1 and 2

    for elem in xml_root.iter():
        # every Cell contains a mapping of en1 -rel(some value)-> en2
        if 'Cell' in elem.tag:
            # follow the order in the oaei rdf mappings
            en1, en2, measure, rel = tuple(list(elem))
            en1, en2 = list(en1.attrib.values())[0], list(en2.attrib.values())[0]
            en1, en2 = reformat_entity_uri(en1, onto1), reformat_entity_uri(en2, onto2)
            measure, rel = measure.text, rel.text
            if rel == "=":
                legal_mappings.append([en1, en2, measure])
            else:
                illegal_mappings.append([en1, en2, measure])

    return legal_mappings, illegal_mappings


def rdf_to_tsv(rdf_file):
    output_dir = rdf_file.replace(".rdf", "")
    legal_mappings, illegal_mappings = read_mappings(rdf_to_xml(rdf_file))
    _df = pd.DataFrame(legal_mappings, columns=["Entity1", "Entity2", "Value"])
    _df.to_csv(output_dir + "_legal.tsv", index=False, sep='\t')
    _df = pd.DataFrame(illegal_mappings, columns=["Entity1", "Entity2", "Value"])
    _df.to_csv(output_dir + "_illegal.tsv", index=False, sep='\t')


# file = main_dir + "../data/references/snomed2nci.rdf"
# rdf_to_tsv(file)
