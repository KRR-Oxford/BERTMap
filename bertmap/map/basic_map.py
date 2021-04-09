"""
The Cross-ontology Mapping superclass, it requires the implementation of mapping computation algorithm. 
"""

import pandas as pd
from bertmap.onto import Ontology
from bertmap.onto import OntoEvaluator
from bertmap.utils import log_print
import time
import seaborn as sns 


class OntoMapping:
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_class2text_tsv, tgt_onto_lexicon_tsv, save_path, 
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
        self.src_onto_class2text_path = src_onto_class2text_tsv
        self.src_onto_class2text = Ontology.load_class2text(self.src_onto_class2text_path)
        self.tgt_onto_class2text_path = tgt_onto_lexicon_tsv
        self.tgt_onto_class2text = Ontology.load_class2text(self.tgt_onto_class2text_path)
        
        # define log print function
        self.log_print = lambda info: log_print(info, f"{self.save_path}/{self.name}.{self.src}2{self.tgt}.log")

        
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

    @staticmethod
    def plot_eval(eval_csv, start_col=0):
        # process eval data
        eval_df = pd.read_csv(eval_csv, index_col=0).iloc[start_col:].reset_index()
        eval_df["Mappings"] = list(eval_df["index"].apply(lambda x: x.split(":")[0]))
        eval_df["Threshold"] = list(eval_df["index"].apply(lambda x: x.split(":")[1]))
        eval_df.drop(columns=["index"], inplace=True)
        eval_df = eval_df.melt(id_vars=["Mappings", "Threshold"], value_vars=["Precision", "Recall", "F1"], var_name="Metric", value_name="Value")
        
        # set styles
        sns.set(style='darkgrid', rc={"font.weight": "bold", "font.size": 20, "axes.labelsize": 20,
                                      "axes.titlesize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
                                      "font.family": "Times New Roman", "axes.labelweight": "bold", 
                                      "axes.titleweight": "bold", "axes.titlepad": 10})
        # create FacetGrid plots for all mappings
        g = sns.FacetGrid(eval_df, col="Mappings", hue="Metric", height=5, aspect=1, margin_titles=True)
        g.map(sns.lineplot, "Threshold", "Value", alpha=.7, marker="o")
        name_mappings = list(eval_df["Mappings"].drop_duplicates())
        for i in range(len(name_mappings)):
            name = name_mappings[i]
            part = eval_df[eval_df["Mappings"] == name]
            part_f1 = part[part["Metric"] == "F1"]
            ax = g.axes.flat[i]
            ax.axvline(part_f1["Threshold"].loc[part_f1["Value"].idxmax()], ls='--',c='r', label="max(F1)")
            ax.legend(loc="upper left")
            ax.set_title(name, color="gray")
        g.fig.suptitle("Plots of Precision, Recall, Macro-F1 against Threshold for Combined, SRC2TGT and TGT2SRC Mappings", y=1.02, fontsize=20, weight="bold")
        return g
