import pandas as pd
from onto_align.onto import Ontology
from onto_align.onto import OntoEvaluator


class OntoExperiment:
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_data_tsv, tgt_data_tsv, task_suffix="small", exp_name="norm_edit_dist"):
        
        # basic information
        self.src = src_onto_iri_abbr  # iri abbreviation of source ontology without ":"
        self.src_iri = Ontology.abbr2iri_dict[self.src+":"]  # full source iri
        self.tgt = tgt_onto_iri_abbr  # iri abbreviation of target ontology without ":"
        self.tgt_iri = Ontology.abbr2iri_dict[self.tgt+":"]  # full target iri
        self.task_suffix = task_suffix  # small or whole
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
    def evaluate(pre_tsv, ref_tsv, except_tsv=None, task_name = "0", threshold=0.0):
        evaluator = OntoEvaluator(pre_tsv, ref_tsv, except_tsv, threshold=threshold)
        result_df = pd.DataFrame(columns=["Precision", "Recall", "F1", "#Illegal", "Threshold"])
        result_df.loc[task_name] = [evaluator.P, evaluator.R, evaluator.F1, evaluator.num_illegal, threshold]
        result_df = result_df.round({"Precision": 3, "Recall": 3, "F1": 3})
        return result_df
        
    @staticmethod
    def interval_split(num_splits, max_num):
        max_range = list(range(max_num))
        interval_range = max_num // num_splits
        for i in range(num_splits + 1):
            start = i * interval_range
            end = min((i + 1) * interval_range, max_num)
            yield max_range[start: end]
    