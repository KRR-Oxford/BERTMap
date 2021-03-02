from owlready2 import get_ontology
import pandas as pd
import xml.etree.ElementTree as ET


class Ontology:

    # for simplification of the ontology uris
    iri_abbr_tsv = __file__.replace("ontology.py", "") + "iri_abbr.tsv"
    iri2abbr_dict = pd.read_csv(iri_abbr_tsv, index_col=0, squeeze=True, sep='\t').to_dict()
    abbr2iri_dict = {v: k for k, v in iri2abbr_dict.items()}

    def __init__(self, onto_file):
        self.onto = get_ontology(f"file://{onto_file}").load()
        self.iri = self.onto.base_iri
        self.iri_abbr = self.iri2abbr_dict[self.iri]
        
    def iri_label_df(self):
        iri_label_df = pd.DataFrame(columns=["entity-iri", "entity-labels-list"])
        iri_list, labels_list = [], []
        for entity in self.onto.classes():
            # lowercase and remove underscores "_", with "<sep>" indicating label boundaries
            labels = " <sep> ".join(list(set(entity.label))).lower().replace("_", " ")
            # print(entity.iri, labels)
            iri_list.append(entity.iri)
            labels_list.append(labels)
        iri_label_df["entity-iri"] = iri_list
        iri_label_df["entity-labels-list"] = labels_list
        return iri_label_df
            
        
    def iri_text_df(self, *lexical_properties):
        """Generate the parallel lists of (iri, textual_info) pairs where:
           textual_info: the merged lexcial information attached to an entity 
        """
        iri_text_df = pd.DataFrame(columns=["entity-iri", "entity-text"])
        iri_list, text_list = [], []
        for entity in self.onto.classes():
            # the lower-cased merged lexical infomation 
            textual_info = " <property> ".join([" <sep> ".join(list(set(getattr(entity, lp)))) for lp in lexical_properties]).lower().replace("_", " ")
            iri_list.append(entity.iri)
            text_list.append(textual_info)
        iri_text_df["entity-iri"] = iri_list
        iri_text_df["entity-text"] = text_list
        return iri_text_df
                
    @classmethod
    def set_iri_abbr_dict(cls, iri_abbr_tsv):
        """ Read the aligned URI-Abbr_URI pairs and form two dictionaries """
        cls.iri_abbr_tsv = iri_abbr_tsv
        cls.iri2abbr_dict = pd.read_csv(iri_abbr_tsv, sep='\t').to_dict()
        cls.abbr2iri_dict = {
            v: k for k, v in cls.iri2abbr_dict.items()
        }

    @classmethod
    def reformat_entity_uri(cls, entity_iri, onto_iri, inverse=False):
        """
        Returns: onto_uri#fragment <=> onto_prefix:fragment
        """
        if not inverse:
            entity_iri = entity_iri.replace(onto_iri, cls.iri2abbr_dict[onto_iri])
        else:
            entity_iri = entity_iri.replace(cls.abbr2iri_dict[onto_iri], onto_iri)
        return entity_iri

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
