"""
OntoText class that handles text data generation from owlready2 Ontology object.
"""

import json
from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, Optional

import pandas as pd
from bertmap.utils import batch_split, uniqify
from owlready2.entity import ThingClass
from owlready2.namespace import Ontology
from owlready2.prop import IndividualValueList


class OntoText():
    
    # one can manually add more full iri - abbreviated iri pairs here
    namespaces = {
        "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#": "fma:",
        "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#": "nci:",
        "http://www.ihtsdo.org/snomed#": "snomed:",
    }
    inv_namespaces = {v: k for k, v in namespaces.items()}
    
    # exclude mistaken parsing of string "null" to NaN
    na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})

    def __init__(self, 
                 onto: Ontology, 
                 iri_abbr: Optional[str]=None, 
                 properties: List[str]=["label"], 
                 classtexts_file: Optional[str]=""):
        
        # load owlready2 ontology and assign attributes
        self.onto = onto
        self.name = self.onto.name
        self.iri = self.onto.base_iri
        self.properties = properties
        
        # get the abbreviated iri for clearer presentation later on
        if self.iri in self.namespaces.keys(): self.iri_abbr = self.namespaces[self.iri]
        elif not iri_abbr: print("Please provide the abbreviated IRI of the input ontology as argument {iri_abbr}.")
        else: self.iri_abbr = iri_abbr
        
        # create or load texts associated to each class
        if not classtexts_file: self.extract_classtexts(*self.properties)
        else: self.load_classtexts(classtexts_file)
        
        # assign indices to classes
        self.class2idx = dict()
        self.idx2class = dict()
        i = 0
        for class_iri, _ in self.texts.items():
            self.class2idx[class_iri] = i
            self.idx2class[i] = class_iri
            i += 1
        
    def __repr__(self):
        iri_abbr = self.iri_abbr.replace(":", "")
        return f"<OntoText abbr='{iri_abbr}' num_classes={len(self.class2idx)} num_texts={self.num_texts} prop={self.properties}>"
        
    def extract_classtexts(self, *properties) -> None:
        """Construct dict(class-iri -> dict(property -> class-text))
        """
        self.num_texts = 0
        self.texts = defaultdict(lambda: defaultdict(list))
        # default lexicon information is the "labels"
        if not properties: properties = ["label"]
        for cl in self.onto.classes():
            cl_iri_abbr = self.abbr_entity_iri(cl.iri)
            for prop in properties:
                # lowercase and remove underscores "_"
                self.texts[cl_iri_abbr][prop] = self.preprocess_classtexts(cl, prop)
                self.num_texts += len(self.texts[cl_iri_abbr][prop])
    
    def save_classtexts(self, classtexts_file: str) -> None:
        # do not sort keys otherwise class2idx and idx2class will be mis-used later
        with open(classtexts_file, "w") as f:
            json.dump(self.texts, f, indent=4, separators=(',', ': '))
    
    def load_classtexts(self, classtexts_file: str) -> None:
        with open(classtexts_file, "r") as f:
            self.texts = json.load(f)
        # compute number of texts
        self.num_texts = 0
        for td in self.texts.values():
            for txts in td.values():
                self.num_texts += len(txts)
        
    def batch_iterator(self, 
                       selected_classes: List[str], 
                       batch_size: int) -> Iterable[Dict[str, Dict]]:
        """
        Args:
            selected_classes (List[str])
            batch_size (int)

        Yields:
            dict: dictionary that stores a batch of (class-iri, class-text) pairs.
        """
        idx_splits = batch_split(batch_size, max_num=len(selected_classes))
        for idxs in idx_splits:
            batch = OrderedDict()
            for i in idxs: batch[selected_classes[i]] = self.texts[selected_classes[i]]
            yield batch
            
    @staticmethod
    def preprocess_classtexts(cl: ThingClass, prop: str) -> List[str]:
        """Preprocessing the texts of a class given by a particular property including
        underscores removal and lower-casing.

        Args:
            cl : class entity
            prop (str): name of the property, e.g. "label"

        Returns:
            list: cleaned and uniqified class-texts
        """
        raw_texts = getattr(cl, prop)
        assert type(raw_texts) is IndividualValueList
        cleaned_texts = [txt.lower().replace("_", " ") for txt in raw_texts]
        return uniqify(cleaned_texts)

    def abbr_entity_iri(self, entity_iri: str) -> str:
        """onto_iri#fragment => onto_prefix:fragment
        """
        return entity_iri.replace(self.iri, self.namespaces[self.iri])
    
    def expand_entity_iri(self, entity_iri_abbr: str) -> str:
        """onto_iri#fragment <= onto_prefix:fragment
        """
        return entity_iri_abbr.replace(self.iri_abbr, self.inv_namespaces[self.iri_abbr])
