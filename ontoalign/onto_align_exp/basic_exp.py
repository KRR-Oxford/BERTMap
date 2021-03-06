import pandas as pd
from ontoalign.onto import Ontology
from ontoalign.onto import OntoEvaluator
from ontoalign.utils import log_print
import sys


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
        
        # data batch generator
        self.src_batch_generator = Ontology.iri_lexicon_batch_generator(src_onto_lexicon_tsv)
        self.tgt_batch_generator = Ontology.iri_lexicon_batch_generator(tgt_onto_lexicon_tsv)
        
        # define log print function
        self.log_print = lambda info: log_print(info, self.save_path)

        
    def run(self):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError
        
    @staticmethod
    def evaluate(pre_tsv, ref_tsv, except_tsv=None, task_name = "0", threshold=0.0):
        evaluator = OntoEvaluator(pre_tsv, ref_tsv, except_tsv, threshold=threshold)
        result_df = pd.DataFrame(columns=["Precision", "Recall", "F1", "#Illegal", "Threshold"])
        result_df.loc[task_name] = [evaluator.P, evaluator.R, evaluator.F1, evaluator.num_illegal, threshold]
        result_df = result_df.round({"Precision": 3, "Recall": 3, "F1": 3})
        return result_df
