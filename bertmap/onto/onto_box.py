"""
OntoBox class that handles data generation from owlready2 Ontology object.
"""

import owlready2
from owlready2 import get_ontology
import pandas as pd
from bertmap.utils import batch_split, uniqify
import math
import json
from collections import defaultdict


class OntoBox():
    
    # one can manually add more full iri - abbreviated iri pairs here
    iri2abbr_dict = {
        "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#": "fma:",
        "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#": "nci:",
        "http://www.ihtsdo.org/snomed#": "snomed:",
    }
    abbr2iri_dict = {v: k for k, v in iri2abbr_dict.items()}
    
    # exclude mistaken parsing of string "null" to NaN
    na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})

    def __init__(self, onto_file, classtexts_file=None, iri_abbr=None, *properties):
        
        # load owlready2 ontology and assign attributes
        self.onto = get_ontology(f"file://{onto_file}").load()
        self.name = self.onto.name
        self.iri = self.onto.base_iri
        
        # get the abbreviated iri for clearer presentation later on
        if self.iri in self.iri2abbr_dict.keys(): self.iri_abbr = self.iri2abbr_dict[self.iri]
        elif not iri_abbr: print("Please provide the abbreviated IRI of the input ontology as argument {iri_abbr}.")
        else: self.iri_abbr = iri_abbr
        
        # create or load texts associated to each class
        if not classtexts_file: self.extract_classtexts(*properties)
        else: self.load_classtexts(classtexts_file)
        
        # assign indices to classes
        self.class2idx = dict()
        self.idx2class = dict()
        i = 0
        for class_iri, _ in self.classtexts.items():
            self.class2idx[class_iri] = i
            self.idx2class[i] = class_iri
            i += 1
        
    def __repr__(self):
        report = f"<OntoBox ontology=\'{self.name}.owl\'>\n"
        iri_abbr = self.iri_abbr.replace(":", "")
        report += f"\t<IRI full=\'{self.iri}\' abbr=\'{iri_abbr}\'>\n"
        report += f"\t<Class num={len(self.class2idx)}>\n"
        report += "</OntoBox>\n"
        return report
        
    def extract_classtexts(self, *properties):
        """Construct dict(class-iri -> dict(property -> class-text))
        """
        self.classtexts = defaultdict(lambda: defaultdict(list))
        # default lexicon information is the "labels"
        if not properties: properties = ["label"]
        for cl in self.onto.classes():
            cl_iri_abbr = self.abbr_entity_iri(cl.iri)
            for prop in properties:
                # lowercase and remove underscores "_"
                self.classtexts[cl_iri_abbr][prop] = self.preprocess_classtext(cl, prop)
    
    def save_classtexts(self, classtext_file):
        with open(classtext_file, "w") as f:
            json.dump(self.classtexts, f, indent=4, separators=(',', ': '), sort_keys=True)
    
    def load_classtexts(self, classtext_file):
        with open(classtext_file, "r") as f:
            self.classtexts = json.load(f)
    
    def batch_iterator(self, batch_size: int):
        """
        Args:
            batch_size (int)

        Yields:
            dict: dictionary that stores a batch of (class-iri, class-text) pairs.
        """
        idx_splits = batch_split(batch_size, max_num=len(self.idx2class))
        for indices in idx_splits:
            yield {self.idx2class[i]: self.classtexts[self.idx2class[i]] for i in indices}
            
    @staticmethod
    def preprocess_classtext(cl, prop):
        """Preprocessing the texts of a class given by a particular property including
        underscores removal and lower-casing.

        Args:
            cl : class entity
            prop (str): name of the property, e.g. "label"

        Returns:
            list: cleaned and uniqified class-texts
        """
        raw_texts = getattr(cl, prop)
        assert type(raw_texts) is owlready2.prop.IndividualValueList
        cleaned_texts = [lexicon.lower().replace("_", " ") for lexicon in raw_texts]
        return uniqify(cleaned_texts)

    def abbr_entity_iri(self, entity_iri):
        """onto_iri#fragment => onto_prefix:fragment"""
        return entity_iri.replace(self.iri, self.iri2abbr_dict[self.iri])
    
    def expand_entity_iri(self, entity_iri_abbr):
        """onto_iri#fragment <= onto_prefix:fragment"""
        return entity_iri_abbr.replace(self.iri_abbr, self.abbr2iri_dict[self.iri_abbr])
    
    @classmethod
    def depth_max(cls, c):
        """Get te maximum depth of a class to the root"""
        supclasses = owlready2.super_classes(c=c)
        if len(supclasses) == 0:
            return 0
        d_max = 0
        for super_c in supclasses:
            super_d = cls.depth_max(c=super_c)
            if super_d > d_max:
                d_max = super_d
        return d_max + 1

    @classmethod
    def depth_min(cls, c):
        """Get te minimum depth of a class to the root"""
        supclasses = owlready2.super_classes(c=c)
        if len(supclasses) == 0:
            return 0
        d_min = math.inf
        for super_c in supclasses:
            super_d = cls.depth_min(c=super_c)
            if super_d < d_min:
                d_min = super_d
        return d_min + 1
