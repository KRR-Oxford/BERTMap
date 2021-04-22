"""
Mapping Generation superclass on using some kind of normalized distance metric or classifier (from fine-tuned BERT):

   Prelimniary Algorithm (One-side-fixed Search):
   
        Compute the *Value between each source-target entity pair where Value is defined by:
           Dist = norm_distance(class1, class2)
           Value = norm_similarity(class1, class2)
           
        [Fix the source side] 
            For each source class (class1), pick the target class (class2) according to the min(Dist) or max(Value)
            
        [Fix the target side] 
            For each target class (class2), pick the source class (class1) according to the min(Dist) opr max(Value)
            
        Remove the duplicates
    
    Note: The search space can be reduced by setting up a candidate selection algorithm.
    Current supported candidate selection: [Subword-level Inverted Index, ]
"""

import pandas as pd
from bertmap.onto import Ontology, OntoEvaluator, OntoInvertedIndex
from bertmap.utils import log_print
from collections import defaultdict
from copy import deepcopy
import time
import seaborn as sns 
import math
import re


class OntoMapping:
    
    def __init__(self, src, tgt, src_class2text_path, tgt_class2text_path, save_path, 
                 task_suffix="undefined_task", name="undefined_exp"):
        
        # basic information
        self.src = src  # iri abbreviation of source ontology without ":"
        self.src_iri = Ontology.abbr2iri_dict[self.src+":"]  # full source iri
        self.tgt = tgt  # iri abbreviation of target ontology without ":"
        self.tgt_iri = Ontology.abbr2iri_dict[self.tgt+":"]  # full target iri
        self.task_suffix = task_suffix  # small or whole
        self.name = name
        self.save_path = save_path
        
        # onto text data
        self.src_onto_class2text_path = src_class2text_path
        self.src_onto_class2text = Ontology.load_class2text(self.src_onto_class2text_path)
        self.tgt_onto_class2text_path = tgt_class2text_path
        self.tgt_onto_class2text = Ontology.load_class2text(self.tgt_onto_class2text_path)
        
        # for candidate selections
        self.src_index = None
        self.tgt_index = None
        self.candidate_limit = 100
        
        # define log print function
        self.log_print = lambda info: log_print(info, f"{self.save_path}/map.log")

    def run(self):
        t_start = time.time()
        # fix SRC side
        self.fixed_one_side_alignment("SRC")
        t_src = time.time()
        self.log_print(f'the program time for computing src2tgt mappings is :{t_src - t_start}')
        # fix TGT side
        self.fixed_one_side_alignment("TGT", start=1282)
        t_tgt= time.time()
        self.log_print(f'the program time for computing tgt2src mappings is :{t_tgt - t_src}')
        t_end = time.time()
        self.log2maps(f"{self.save_path}/map.log", keep=1)
        self.log_print(f'the overall program time is :{t_end - t_start}')
        
    def align_config(self, flag="SRC"):
        """Configurations for swithcing the fixed ontology side."""
        raise NotImplementedError
    
    def fixed_one_side_alignment(self, flag="SRC"):
        raise NotImplementedError
    
    def select_candidates(self, class_text, flag="SRC"):
        assert flag == "SRC" or flag == "TGT"
        onto_index = self.tgt_index if flag == "SRC" else self.src_index
        class2text = self.tgt_onto_class2text if flag == "SRC" else self.src_onto_class2text
        candidates = defaultdict(lambda: 0)
        tokens = onto_index.tokenize_class_text(class_text)
        D = len(class2text)  # num of "documents" (classes)
        for tk in tokens:
            sharing_classes = onto_index.index[tk]
            if not sharing_classes:
                continue
            # We use idf instead of tf because the text for each class is of different length, tf is not a fair measure
            idf = math.log10(D / len(sharing_classes))
            for class_id in sharing_classes:
                candidates[class_id] += idf
        candidates = list(sorted(candidates.items(), key=lambda item: item[1], reverse=True))[:self.candidate_limit]
        print(f"Select {len(candidates)} candidates ...")
        return class2text.iloc[[c[0] for c in candidates]].reset_index(drop=True)
    
    def set_inverted_index(self, flag="SRC", tokenizer_path="emilyalsentzer/Bio_ClinicalBERT", cut=0, clear=True):
        assert flag == "SRC" or flag == "TGT"
        onto_index = OntoInvertedIndex(tokenizer_path)
        onto_name = self.src if flag == "SRC" else self.tgt
        onto_class2text = self.src_onto_class2text if flag == "SRC" else self.tgt_onto_class2text
        onto_index.construct_index(onto_name, onto_class2text, cut=cut, clear=clear)
        index_name = f"{flag.lower()}_index"
        setattr(self, index_name, deepcopy(onto_index))
        
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
    
    @staticmethod
    def log2maps(log_path, keep=1):
        with open(log_path, "r") as f:
            lines = f.readlines()
        src_maps = []
        tgt_maps = []
        src_pa = r"\[SRC:.*Mapping: \['(.+)', '(.+)', (.+)\]\]"
        tgt_pa = r"\[TGT:.*Mapping: \['(.+)', '(.+)', (.+)\]\]"
        # tgt_pa2 = r"\[nci2fma.*Mapping: \['(.+)', '(.+)', (.+)\]\]"
        record = {"src": (None, 0), "tgt": (None, 0)}
        for line in lines:
            if re.findall(src_pa, line):
                map = re.findall(src_pa, line)[0]
                if (not map[0] == record["src"][0]) or record["src"][1] < keep:
                    record["src"] = (map[0], record["src"][1] + 1)
                    src_maps.append(map)
            elif re.findall(tgt_pa, line):
                map = re.findall(tgt_pa, line)[0]
                if (not map[1] == record["tgt"][0]) or record["tgt"][1] < keep:
                    record["tgt"] = (map[1], record["tgt"][1] + 1)
                    tgt_maps.append(map)
        
        save_path = "/".join(log_path.split("/")[:-1])
        # even_maps = [maps[i] for i in range(0, len(maps), 2)]
        src_df = pd.DataFrame(src_maps, columns=["Entity1", "Entity2", "Value"])
        src_df.to_csv(f"{save_path}/src2tgt.maps.tsv", index=False, sep='\t')

        tgt_df = pd.DataFrame(tgt_maps, columns=["Entity1", "Entity2", "Value"])
        tgt_df.to_csv(f"{save_path}/tgt2src.maps.tsv", index=False, sep='\t')
        
        combined_df = src_df.append(tgt_df).drop_duplicates().dropna()
        combined_df.to_csv(f"{save_path}/combined.maps.tsv", index=False, sep='\t')
