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

from os import system
import re
import sys
import time
from collections import defaultdict
from typing import Optional, Tuple, Union

import pandas as pd
import seaborn as sns
from bertmap.onto import OntoBox, OntoEvaluator
from bertmap.utils import banner, log_print
from pandas import DataFrame
import numpy as np


class OntoMapping:
    def __init__(
        self,
        src_ob: OntoBox,
        tgt_ob: OntoBox,
        candidate_limit: Optional[int] = 50,
        save_dir: str = "",
    ):

        self.src_ob = src_ob
        self.tgt_ob = tgt_ob
        self.candidate_limit = candidate_limit
        self.save_dir = save_dir
        self.log_print = lambda info: log_print(
            info, f"{self.save_dir}/map.{self.candidate_limit}.log"
        )
        self.start_time = None

    def run(self) -> None:
        """Run the computation of mapping calculation"""
        t_start = time.time()
        self.log_print(f"Candidate Limit: {self.candidate_limit}")
        self.alignment("SRC")
        t_src = time.time()
        self.log_print(f"the program time for computing src2tgt mappings is :{t_src - t_start}")
        self.alignment("TGT")
        t_tgt = time.time()
        self.log_print(f"the program time for computing tgt2src mappings is :{t_tgt - t_src}")
        t_end = time.time()
        self.log_print(f"the overall program time is :{t_end - t_start}")

    def from_to_config(self, flag: str = "SRC") -> Tuple[OntoBox, OntoBox]:
        """switch source and target OntoBox objects according to flag"""
        assert flag == "SRC" or flag == "TGT"
        if flag == "SRC":
            from_ob, to_ob = self.src_ob, self.tgt_ob
        else:
            from_ob, to_ob = self.tgt_ob, self.src_ob
        return from_ob, to_ob

    def alignment(self, flag: str = "SRC") -> None:
        """Fixed one-side ontology alignment"""
        raise NotImplementedError

    @staticmethod
    def evaluate(
        pre_tsv: Union[str, DataFrame],
        ref_tsv: Union[str, DataFrame],
        ref_ignored_tsv: Optional[Union[str, DataFrame]] = None,
        threshold: float = 0.0,
        prefix: str = "",
    ) -> DataFrame:
        """evaluate the alignment results

        Args:
            pre_tsv (Union[str, DataFrame]): prediction tsv file
            ref_tsv (Union[str, DataFrame]): reference tsv file
            ref_ignored_tsv (Optional[Union[str, DataFrame]], optional): mappings to be ignored during evaluation. Defaults to None.
            threshold (float, optional): mappings with scores lower than threshold will be filtered. Defaults to 0.0.
            prefix (str, optional): index name. Defaults to "".

        Returns:
            DataFrame: evaluation results dataframe
        """
        evaluator = OntoEvaluator(pre_tsv, ref_tsv, ref_ignored_tsv, threshold=threshold)
        result_df = pd.DataFrame(columns=["#Mappings", "#Ignored", "Precision", "Recall", "F1"])
        prefix = prefix + ":" + str(threshold)
        result_df.loc[prefix] = [
            len(evaluator.pre),
            evaluator.num_ignored,
            evaluator.P,
            evaluator.R,
            evaluator.F1,
        ]
        result_df = result_df.round({"Precision": 3, "Recall": 3, "F1": 3})
        if evaluator.id_maps:
            print(
                f"{len(evaluator.id_maps)} mappings ignored are identity mappings (same entity iris on both sides)."
            )
        return result_df

    @staticmethod
    def read_mappings_from_log(log_path: str, keep: int = 1):
        """Read mappings from the mapping computation log"""
        with open(log_path, "r") as f:
            lines = f.readlines()
        src_maps = defaultdict(list)
        tgt_maps = defaultdict(list)
        src_pa = r"\[SRC:.*Mapping: [\(|\[]'(.+)', '(.+)', (.+)[\)|\]]\]"
        tgt_pa = r"\[TGT:.*Mapping: [\(|\[]'(.+)', '(.+)', (.+)[\)|\]]\]"
        for line in lines:
            if re.findall(src_pa, line):
                src_class, tgt_class, value = re.findall(src_pa, line)[0]
                src_maps[src_class].append((tgt_class, value))
                src_maps[src_class].sort(key=lambda x: x[1], reverse=True)
            elif re.findall(tgt_pa, line):
                tgt_class, src_class, value = re.findall(tgt_pa, line)[0]
                tgt_maps[tgt_class].append((src_class, value))
                tgt_maps[tgt_class].sort(key=lambda x: x[1], reverse=True)
        src_maps_kept = []
        tgt_maps_kept = []
        for src_class, v in src_maps.items():
            for tgt_class, value in v[:keep]:
                src_maps_kept.append((src_class, tgt_class, value))
        for tgt_class, v in tgt_maps.items():
            for src_class, value in v[:keep]:
                tgt_maps_kept.append((src_class, tgt_class, value))
        src_df = pd.DataFrame(src_maps_kept, columns=["Entity1", "Entity2", "Value"])
        tgt_df = pd.DataFrame(tgt_maps_kept, columns=["Entity1", "Entity2", "Value"])
        combined_df = src_df.append(tgt_df).drop_duplicates().dropna()
        return src_df, tgt_df, combined_df

    @staticmethod
    def print_eval(eval_csv: str, comment: str=""):
        """Print best string match and system results for easier recording"""
        df = pd.read_csv(eval_csv, index_col=0)
        best_string_match_idx = df["F1"][-3:].idxmax()
        best_system_idx = df["F1"][:-3].idxmax()
        banner(f"Evaluation results for mapping at {eval_csv}", sym="#")
        banner(f"Best string match results {comment}")
        P = df.loc[best_string_match_idx]["Precision"]
        R = df.loc[best_string_match_idx]["Recall"]
        F1 = df.loc[best_string_match_idx]["F1"]
        print(
            f"P={round(P, 3)}",
            f"R={round(R, 3)}",
            f"F1={round(F1, 3)}",
            str(best_string_match_idx).replace("combined", "cb"),
        )
        banner(f"Best BERTMap results {comment}")
        P = df.loc[best_system_idx]["Precision"]
        R = df.loc[best_system_idx]["Recall"]
        F1 = df.loc[best_system_idx]["F1"]
        print(
            f"P={round(P, 3)}",
            f"R={round(R, 3)}",
            f"F1={round(F1, 3)}",
            str(best_system_idx).replace("combined", "cb"),
        )

    @staticmethod
    def print_eval_latex(eval_csv: str):
        """Print best string match and system results for easier recording"""
        df = pd.read_csv(eval_csv, index_col=0)
        best_string_match_idx = df["F1"][-3:].idxmax()
        best_system_idx = df["F1"][:-3].idxmax()
        P_str = df.loc[best_string_match_idx]["Precision"]
        R_str = df.loc[best_string_match_idx]["Recall"]
        F1_str = df.loc[best_string_match_idx]["F1"]
        string_match = (
            f"{best_string_match_idx} & {round(P_str, 3)} & {round(R_str, 3)} & {round(F1_str, 3)}"
        )
        P = df.loc[best_system_idx]["Precision"]
        R = df.loc[best_system_idx]["Recall"]
        F1 = df.loc[best_system_idx]["F1"]
        system = f"{best_system_idx} & {round(P, 3)} & {round(R, 3)} & {round(F1, 3)}"
        return string_match, system

    @staticmethod
    def plot_eval(eval_csv, start_col=0):
        """plot the evaluation results (under review)"""
        # process eval data
        eval_df = pd.read_csv(eval_csv, index_col=0).iloc[start_col:].reset_index()
        eval_df["Mappings"] = list(
            eval_df["index"].apply(
                lambda x: x.split(":")[0].replace("src", "src2").replace("tgt", "tgt2src").replace("src2", "src2tgt")
            )
        )
        eval_df["Threshold"] = list(eval_df["index"].apply(lambda x: x.split(":")[1]))
        eval_df.drop(columns=["index"], inplace=True)
        eval_df = eval_df.melt(
            id_vars=["Mappings", "Threshold"],
            value_vars=["Precision", "Recall", "F1"],
            var_name="Metric",
            value_name="Value",
        )

        # set styles
        sns.set(
            style="darkgrid",
            rc={
                "font.weight": "bold",
                "font.size": 15,
                "axes.labelsize": 13,
                "axes.titlesize": 13,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "font.family": "Times New Roman",
                "axes.labelweight": "bold",
                "axes.titleweight": "bold",
                "axes.titlepad": 15,
            },
        )
        # create FacetGrid plots for all mappings
        g = sns.FacetGrid(
            eval_df, col="Mappings", hue="Metric", height=5, aspect=1, margin_titles=True
        )
        g.map(sns.lineplot, "Threshold", "Value", alpha=0.7, marker=None)
        name_mappings = list(eval_df["Mappings"].drop_duplicates())
        for i in range(len(name_mappings)):
            name = name_mappings[i]
            part = eval_df[eval_df["Mappings"] == name]
            part_f1 = part[part["Metric"] == "F1"]
            ax = g.axes.flat[i]
            ax.axvline(
                part_f1["Threshold"].loc[part_f1["Value"].idxmax()], ls="--", c="r", label="max(F1)"
            )
            ax.legend(loc="upper left")
            ax.set_title(name, color="gray")
        # g.fig.suptitle(
        #     "Plots of Precision, Recall, Macro-F1 against Threshold for Combined, SRC2TGT and TGT2SRC Mappings",
        #     y=1.02,
        #     fontsize=12,
        #     # weight="bold",
        # )
        g.set(xticks=eval_df["Threshold"].iloc[[0, 50, 100, 150, 200, -1]])
        return g
