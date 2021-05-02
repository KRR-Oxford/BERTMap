"""
OntoBox class that handles data generation from owlready2 Ontology object.
"""

import owlready2
from owlready2 import get_ontology
from bertmap.onto import OntoText
from bertmap.onto import OntoInvertedIndex
import math, os, re, ast
from typing import Optional, List
from shutil import copy2
from pathlib import Path


class OntoBox():

    def __init__(self, 
                 onto_file: str, 
                 onto_iri_abbr: Optional[str]=None, 
                 textual_properties: List[str]=["label"],
                 tokenizer_path: str="emilyalsentzer/Bio_ClinicalBERT", 
                 cut: int=0,
                 from_saved=False):
        
        # load owlready2 ontology and assign attributes
        self.onto_file = onto_file
        self.onto = get_ontology(f"file://{onto_file}").load()
        if not from_saved:
            self.onto_text = OntoText(self.onto, iri_abbr=onto_iri_abbr, properties=textual_properties)
            self.onto_index = OntoInvertedIndex(self.onto_text, tokenizer_path, cut=cut)
        else: pass  # construct OntoText and ontoIndex from saved files
        
    def __repr__(self):
        report = f"<OntoBox> onto='{self.onto.name}.owl' iri='{self.onto.base_iri}'>\n"
        report += f"\t{self.onto_text}" + "\n"
        report += f"\t{self.onto_index}" + "\n"
        report += "</OntoBox>\n"
        return report
    
    def save(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        copy2(self.onto_file, save_dir)
        self.onto_text.save_classtexts(save_dir + f"/{self.onto.name}.ctxt.json")
        self.onto_index.save_index(save_dir + f"/{self.onto.name}.ind.json")
        with open(save_dir + "/info.txt", "w") as f: f.write(str(self))
    
    @classmethod
    def from_saved(cls, save_dir):
        
        # check and load onto data files
        onto_file = []
        classtexts_file = []
        inv_index_file = []
        info_file = []
        for file in os.listdir(save_dir):
            if file.endswith(".owl"): onto_file.append(file)
            elif file.endswith(".ctxt.json"): classtexts_file.append(file)
            elif file.endswith(".ind.json"): inv_index_file.append(file)
            elif file == "info.txt": info_file.append(file)
            else: print(f"[ERROR] invalid file detected: {file}"); return
        if len(onto_file) != 1 or len(classtexts_file) != 1 or len(inv_index_file) != 1: 
            print(f"[ERROR] multiple data files detected"); return
        with open(f"{save_dir}/{info_file[0]}", "r") as f: 
            lines = f.readlines()
            iri_abbr = re.findall(r"iri=\'(.+)\'", lines[0])[0]
            properties = ast.literal_eval(re.findall(r"prop=(\[.+\])", lines[1])[0])
            cut = int(re.findall(r"cut=([0-9]+)", lines[2])[0])    
        # construct the OntoBox instance   
        print(f"found files of correct formats, trying to load ontology data from {save_dir}")
        ontobox = cls(onto_file=f"{save_dir}/{onto_file[0]}", from_saved=True)
        ontobox.onto_text = OntoText(ontobox.onto, iri_abbr, properties, f"{save_dir}/{classtexts_file[0]}")
        ontobox.onto_index = OntoInvertedIndex(cut=cut, index_file=f"{save_dir}/{inv_index_file[0]}")
        return ontobox
    
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
