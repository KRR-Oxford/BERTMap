import pandas as pd
from ontoalign.onto import Ontology
from ontoalign.onto import OntoEvaluator
from ontoalign.utils import log_print
import time


class OntoAlignExperiment:
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, save_path, 
                 task_suffix="undefined_task", name="undefined_exp"):
        
        # basic information
        self.src = src_onto_iri_abbr  # iri abbreviation of source ontology without ":"
        self.src_iri = Ontology.abbr2iri_dict[self.src+":"]  # full source iri
        self.tgt = tgt_onto_iri_abbr  # iri abbreviation of target ontology without ":"
        self.tgt_iri = Ontology.abbr2iri_dict[self.tgt+":"]  # full target iri
        self.task_suffix = task_suffix  # small or whole
        self.name = name
        self.save_path = save_path
        
        # onto lexicon data
        self.src_onto_lexicon_path = src_onto_lexicon_tsv
        self.src_onto_lexicon = Ontology.load_iri_lexicon_file(self.src_onto_lexicon_path)
        self.tgt_onto_lexicon_path = tgt_onto_lexicon_tsv
        self.tgt_onto_lexicon = Ontology.load_iri_lexicon_file(self.tgt_onto_lexicon_path)
        
        # define log print function
        self.log_print = lambda info: log_print(info, f"{self.save_path}/{self.name}.log")

        
    def run(self):
        t_start = time.time()
        self.alignment()
        t_end = time.time()
        t = t_end - t_start
        self.log_print('the program time is :%s' %t)
    
    def alignment(self):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError
        
    @staticmethod
    def evaluate(pre_tsv, ref_tsv, except_tsv=None, task_name = "0", threshold=0.0):
        evaluator = OntoEvaluator(pre_tsv, ref_tsv, except_tsv, threshold=threshold)
        result_df = pd.DataFrame(columns=["Precision", "Recall", "F1", "#Illegal"])
        task_name = task_name + ":" + str(threshold)
        result_df.loc[task_name] = [evaluator.P, evaluator.R, evaluator.F1, evaluator.num_illegal]
        result_df = result_df.round({"Precision": 3, "Recall": 3, "F1": 3})
        return result_df
