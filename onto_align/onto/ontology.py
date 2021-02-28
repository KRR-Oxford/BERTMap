from owlready2 import get_ontology
import pandas as pd
import xml.etree.ElementTree as ET


class Ontology:

    # for simplification of the ontology uris
    iri_abbr_tsv = __file__.replace("ontology.py", "") + "iri_abbr.tsv"
    iri2abbr = pd.read_csv(iri_abbr_tsv, index_col=0, squeeze=True, sep='\t').to_dict()
    abbr2iri = {v: k for k, v in iri2abbr.items()}

    def __init__(self, onto_file):
        self.onto = get_ontology(f"file://{onto_file}").load()
        self.iri = self.onto.base_iri
        self.iri_abbr = self.iri2abbr[self.iri]

    @classmethod
    def set_iri_abbr_dict(cls, iri_abbr_tsv):
        """ Read the aligned URI-Abbr_URI pairs and form two dictionaries """
        cls.iri_abbr_tsv = iri_abbr_tsv
        cls.iri2abbr = pd.read_csv(iri_abbr_tsv, sep='\t').to_dict()
        cls.abbr2iri = {
            v: k for k, v in cls.iri2abbr.items()
        }

    @classmethod
    def reformat_entity_uri(cls, entity_uri, onto_uri, inverse=False):
        """
        Returns: onto_uri#fragment <=> onto_prefix:fragment
        """
        if not inverse:
            entity_uri = entity_uri.replace(onto_uri, cls.iri2abbr[onto_uri])
        else:
            entity_uri = entity_uri.replace(cls.abbr2iri[onto_uri], onto_uri)
        return entity_uri

    @staticmethod
    def read_onto_uris_from_rdf(rdf_file, *uri_tags):
        """Read the uri of the ontology from the rdf file"""
        xml_root = ET.parse(rdf_file).getroot()
        onto_uris = dict()
        for elem in xml_root.iter():
            for tag in uri_tags:
                if tag in elem.tag:
                    onto_uris[tag] = elem.text
            # end case when all ontologies are captured
            if len(onto_uris) == len(uri_tags):
                break
        return onto_uris
