"""
The Ontology class that handles data generation from owlready2 Ontology object.
"""


from owlready2 import get_ontology
import owlready2
import pandas as pd
import xml.etree.ElementTree as ET
from bertmap.utils import batch_split
import itertools
import math

class Ontology:

    # for simplification of the ontology uris
    onto_iri_abbr_tsv = __file__.replace("ontology.py", "") + "iri_abbr.tsv"
    iri2abbr_dict = pd.read_csv(onto_iri_abbr_tsv, index_col=0, squeeze=True, sep='\t').to_dict()
    abbr2iri_dict = {v: k for k, v in iri2abbr_dict.items()}
    # exclude mistaken parsing of string "null" to NaN
    na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL','null'})

    def __init__(self, onto_file):
        self.onto = get_ontology(f"file://{onto_file}").load()
        self.iri = self.onto.base_iri
        self.iri_abbr = self.iri2abbr_dict[self.iri]
        
    def create_class2text(self, *textual_properties):
        
        # default lexicon information is the "labels"
        if not textual_properties:
            textual_properties = ["label"]
            
        class2text_df = pd.DataFrame()
        iri_list, lexicon_list = [], []
        
        for entity in self.onto.classes():
            # lowercase and remove underscores "_", with "<sep>" indicating label boundaries, "<property" indicating property boundaries
            lexicon = " <property> ".join([self.encode_class_text(entity, lp) for lp in textual_properties])
            # print(entity.iri, labels)
            iri_list.append(self.abbr_entity_iri(entity.iri))
            lexicon_list.append(lexicon)
            
        class2text_df["Class-IRI"] = iri_list
        class2text_df["Class-Text"] = lexicon_list
        return class2text_df
    
    @classmethod
    def load_class2text(cls, class2text_file: str):
        return pd.read_csv(class2text_file, sep="\t", na_values=cls.na_vals, keep_default_na=False)
    
    @classmethod
    def class2text_batch_generator(cls, class2text_file: str, batch_size: int):
        class2text_df = cls.load_class2text(class2text_file)
        index_splits = batch_split(batch_size, max_num=len(class2text_df))
        for split in index_splits:
            yield class2text_df.iloc[split]
            
    @staticmethod
    def encode_class_text(class_entity, textual_property):
        raw_text_list = getattr(class_entity, textual_property)
        assert type(raw_text_list) is owlready2.prop.IndividualValueList
        text_list = [lexicon.lower().replace("_", " ") for lexicon in raw_text_list]
        text_list = list(dict.fromkeys(text_list))  # remove duplicates
        return " <sep> ".join(text_list)
           
    @staticmethod
    def parse_class_text(class_text):
        properties = class_text.split(" <property> ")
        lexicon = itertools.chain.from_iterable([property.split(" <sep> ") for property in properties])
        lexicon = list(filter(lambda x: x != "", lexicon))   # remove the empty lexicon info
        return lexicon, len(lexicon)
        
    @classmethod
    def set_iri_abbr_dict(cls, iri_abbr_tsv):
        """ Read the aligned URI-Abbr_URI pairs and form two dictionaries """
        cls.onto_iri_abbr_tsv = iri_abbr_tsv
        cls.iri2abbr_dict = pd.read_csv(iri_abbr_tsv, sep='\t').to_dict()
        cls.abbr2iri_dict = {
            v: k for k, v in cls.iri2abbr_dict.items()
        }


    def abbr_entity_iri(self, entity_iri):
        """Returns: onto_uri#fragment => onto_prefix:fragment"""
        return entity_iri.replace(self.iri, self.iri2abbr_dict[self.iri])
    
    def expand_entity_iri(self, entity_iri_abbr):
        """Returns: onto_uri#fragment <= onto_prefix:fragment"""
        return entity_iri_abbr.replace(self.iri_abbr, self.abbr2iri_dict[self.iri_abbr])

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
    

    # Get te maximum depth of a class to the root
    @classmethod
    def depth_max(cls, c):
        supclasses = owlready2.super_classes(c=c)
        if len(supclasses) == 0:
            return 0
        d_max = 0
        for super_c in supclasses:
            super_d = cls.depth_max(c=super_c)
            if super_d > d_max:
                d_max = super_d
        return d_max + 1


    # Get te minimum depth of a class to the root
    @classmethod
    def depth_min(cls, c):
        supclasses = owlready2.super_classes(c=c)
        if len(supclasses) == 0:
            return 0
        d_min = math.inf
        for super_c in supclasses:
            super_d = cls.depth_min(c=super_c)
            if super_d < d_min:
                d_min = super_d
        return d_min + 1
