import pandas as pd
from onto_align.onto import Ontology
from onto_align.onto import OntoEvaluator


class OntoExperiment:
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_data_tsv, tgt_data_tsv, task="small", exp_name="norm_edit_dist"):
        
        # basic information
        self.src = src_onto_iri_abbr  # iri abbreviation of source ontology without ":"
        self.src_iri = Ontology.abbr2iri_dict[self.src+":"]  # full source iri
        self.tgt = tgt_onto_iri_abbr  # iri abbreviation of target ontology without ":"
        self.tgt_iri = Ontology.abbr2iri_dict[self.tgt+":"]  # full target iri
        self.task = task  # small or whole
        self.exp_name = exp_name
        
        # data file
        na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL','null'})  # exclude mistaken parsing of string "null" to NaN
        self.src_tsv = pd.read_csv(src_data_tsv, sep="\t", na_values=na_vals, keep_default_na=False)
        self.tgt_tsv = pd.read_csv(tgt_data_tsv, sep='\t', na_values=na_vals, keep_default_na=False)

        
    def run(self):
        raise NotImplementedError
    
    def save(self, save_path):
        raise NotImplementedError
        
    @staticmethod
    def evaluate(pre_tsv, ref_tsv, except_tsv=None):
        evaluator = OntoEvaluator(pre_tsv, ref_tsv, except_tsv)
        print("------ Results ------")
        print("P =", evaluator.P)
        print("R =", evaluator.R)
        print("F1 =", evaluator.F1)
        
    @staticmethod
    def interval_split(num_splits, max_num):
        max_range = list(range(max_num))
        interval_range = max_num // num_splits
        for i in range(num_splits + 1):
            start = i * interval_range
            end = min((i + 1) * interval_range, max_num)
            yield max_range[start: end]
    