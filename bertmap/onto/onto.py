from owlready2 import get_ontology
import owlready2
import pandas as pd
import xml.etree.ElementTree as ET
from bertmap.utils import batch_split
import itertools
import re

class Ontology:

    # for simplification of the ontology uris
    iri_abbr_tsv = __file__.replace("onto.py", "") + "iri_abbr.tsv"
    iri2abbr_dict = pd.read_csv(iri_abbr_tsv, index_col=0, squeeze=True, sep='\t').to_dict()
    abbr2iri_dict = {v: k for k, v in iri2abbr_dict.items()}
    # exclude mistaken parsing of string "null" to NaN
    na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL','null'})

    def __init__(self, onto_file):
        self.onto = get_ontology(f"file://{onto_file}").load()
        self.iri = self.onto.base_iri
        self.iri_abbr = self.iri2abbr_dict[self.iri]
        
    def iri_lexicon_df(self, *lexical_properties):
        
        # default lexicon information is the "labels"
        if not lexical_properties:
            lexical_properties = ["label"]
            
        iri_lexicon_df = pd.DataFrame()
        iri_list, lexicon_list = [], []
        
        for entity in self.onto.classes():
            # lowercase and remove underscores "_", with "<sep>" indicating label boundaries, "<property" indicating property boundaries
            lexicon = " <property> ".join([self.encode_entity_lexicon(entity, lp) for lp in lexical_properties])
            # print(entity.iri, labels)
            iri_list.append(self.abbr_entity_iri(entity.iri))
            lexicon_list.append(lexicon)
            
        iri_lexicon_df["Entity-IRI"] = iri_list
        iri_lexicon_df["Entity-Lexicon"] = lexicon_list
        return iri_lexicon_df
    
    @classmethod
    def load_iri_lexicon_file(cls, iri_lexicon_file: str):
        return pd.read_csv(iri_lexicon_file, sep="\t", na_values=cls.na_vals, keep_default_na=False)
    
    @classmethod
    def iri_lexicon_batch_generator(cls, iri_lexicon_file: str, batch_size: int):
        iri_lexicon_df = cls.load_iri_lexicon_file(iri_lexicon_file)
        index_splits = batch_split(batch_size, max_num=len(iri_lexicon_df))
        for split in index_splits:
            yield iri_lexicon_df.iloc[split]
            
    @staticmethod
    def encode_entity_lexicon(entity, lexical_property):
        raw_lexicon_list = getattr(entity, lexical_property)
        assert type(raw_lexicon_list) is owlready2.prop.IndividualValueList
        lexicon_list = [lexicon.lower().replace("_", " ") for lexicon in raw_lexicon_list]
        lexicon_list = list(dict.fromkeys(lexicon_list))  # remove duplicates
        return " <sep> ".join(lexicon_list)
        
            
    @staticmethod
    def parse_entity_lexicon(entity_lexicon):
        properties = entity_lexicon.split(" <property> ")
        lexicon = itertools.chain.from_iterable([property.split(" <sep> ") for property in properties])
        lexicon = list(filter(lambda x: x != "", lexicon))   # remove the empty lexicon info
        return lexicon, len(lexicon)
        
    @classmethod
    def set_iri_abbr_dict(cls, iri_abbr_tsv):
        """ Read the aligned URI-Abbr_URI pairs and form two dictionaries """
        cls.iri_abbr_tsv = iri_abbr_tsv
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
