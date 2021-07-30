"""Script for evaluating different BERTMap systems as follows:

    1. select hyperparameters (mapping threshold) on the validation set (10% of the ref mappings)
    2. use such threshold to generate the test-set result (90% refs for unsupervised and 70% refs for semi-supervised)

"""

# append the paths
import os

main_dir = os.getcwd().split("BERTMap")[0] + "BERTMap"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable huggingface tokenizer paralellism
import sys

sys.path.append(main_dir)

# import essentials
import argparse
import json
from shutil import copy2
from pathlib import Path
from copy import deepcopy
from transformers import TrainingArguments
import torch
import multiprocessing_on_dill
import pandas as pd
import time

# import bertmap
from bertmap.utils import evenly_divide, set_seed, equal_split, banner
from bertmap.onto import OntoBox
from bertmap.corpora import OntoAlignCorpora
from bertmap.bert import BERTTrainer
from bertmap.map import *

na_vals = pd.io.parsers.STR_NA_VALUES.difference({"NULL", "null", "n/a"})

def validate_maps(config, mode):

    task_dir = config["data"]["task_dir"]

    if mode == "bertmap":
        fine_tune_params = config["fine-tune"]
        learn = fine_tune_params["learning"]
        assert learn == "us" or learn == "ss"
        include_ids = fine_tune_params["include_ids"]
        banner(f"evaluate fine-tuned models of {learn} settings", sym="#")
        exp_dir = (
            task_dir + f"/fine-tune.exp/{learn}.exp" if not include_ids else task_dir + f"/fine-tune.exp/{learn}.ids.exp"
        )


    map_params = deepcopy(config["map"])
    limits = map_params["candidate_limits"]
    del map_params["candidate_limits"]
    for candidate_limit in limits:
        eval_maps(config=config, candidate_limit=candidate_limit)


def eval_maps(config, candidate_limit: int, semi_supervised=False):

    eval_file = f"{exp_dir}/map.{candidate_limit}/eval.{candidate_limit}.csv"
    # In semi-supervised setting, besides considering all the mappings, we should also test it on the test mappings
    if semi_supervised:
        eval_file = f"{exp_dir}/map.{candidate_limit}/eval.{candidate_limit}.test.csv"

    if os.path.exists(eval_file):
        print(f"skip map evaluation for candidate limit {candidate_limit} as existed ...")
        return

    report = pd.DataFrame(columns=["#Mappings", "#Ignored", "Precision", "Recall", "F1"])
    ref = f"{task_dir}/refs/maps.ref.us.tsv"
    ref_ignored = f"{task_dir}/refs/maps.ignored.tsv" if config["corpora"]["ignored_mappings_file"] else None
    # for semi-supervised setting, we should ignore all the mappings in th train an val splits
    if semi_supervised:
        ss_ignored = f"{task_dir}/refs/maps.ignored.ss.tsv"
        if os.path.exists(ss_ignored):
            ref_ignored = ss_ignored
        else:
            train_maps_df = pd.read_csv(
                f"{task_dir}/refs/maps.ref.ss.train.tsv", sep="\t", na_values=na_vals, keep_default_na=False
            )
            val_maps_df = pd.read_csv(
                f"{task_dir}/refs/maps.ref.ss.val.tsv", sep="\t", na_values=na_vals, keep_default_na=False
            )
            ref = f"{task_dir}/refs/maps.ref.ss.test.tsv"
            if ref_ignored:
                ref_ignored = pd.read_csv(ref_ignored, sep="\t", na_values=na_vals, keep_default_na=False)
            else:
                ref_ignored = pd.DataFrame(columns=["Entity1", "Entity2", "Value"])
            ref_ignored = ref_ignored.append(train_maps_df).append(val_maps_df).reset_index(drop=True)
            ref_ignored.to_csv(f"{task_dir}/refs/maps.ignored.ss.tsv", sep="\t", index=False)

    pool = multiprocessing_on_dill.Pool(10)
    eval_results = []
    thresholds = evenly_divide(0, 0.5, 5) + evenly_divide(0.7, 0.94, 24) + evenly_divide(0.95, 1.0, 50)
    for threshold in thresholds:
        threshold = round(threshold, 6)
        eval_results.append(
            pool.apply_async(
                OntoMapping.evaluate,
                args=(
                    f"{exp_dir}/map.{candidate_limit}/combined.{candidate_limit}.tsv",
                    ref,
                    ref_ignored,
                    threshold,
                    f"combined",
                ),
            )
        )
        eval_results.append(
            pool.apply_async(
                OntoMapping.evaluate,
                args=(
                    f"{exp_dir}/map.{candidate_limit}/src.{candidate_limit}.tsv",
                    ref,
                    ref_ignored,
                    threshold,
                    f"src",
                ),
            )
        )
        eval_results.append(
            pool.apply_async(
                OntoMapping.evaluate,
                args=(
                    f"{exp_dir}/map.{candidate_limit}/tgt.{candidate_limit}.tsv",
                    ref,
                    ref_ignored,
                    threshold,
                    f"tgt",
                ),
            )
        )
    pool.close()
    pool.join()

    for result in eval_results:
        result = result.get()
        report = report.append(result)
    print(report)
    report.to_csv(eval_file)
    max_scores = list(report.max()[["Precision", "Recall", "F1"]])
    max_inds = list(report.idxmax()[["Precision", "Recall", "F1"]])
    print(
        f"Best results are: P: {max_scores[0]} ({max_inds[0]}); R: {max_scores[1]} ({max_inds[1]}); F1: {max_scores[2]} ({max_inds[2]})."
    )

if __name__ == "__main__":

    set_seed(888)

    # parse configuration file and specify mode
    parser = argparse.ArgumentParser(description="run bertmap system")
    parser.add_argument("-c", "--config", type=str, help="configuration file for bertmap system", required=True)
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices={"bertmap", "bertembeds", "edit"},
        default="bertmap",
        help="preprocessing data (pre), training BERT model (train), or computing the mappings and evaluate them (map)",
    )
    args = parser.parse_args()

    banner("load configurations", sym="#")
    print(f"configuration-file: {args.config}")
    print(f"mode: {args.mode}")
    with open(args.config, "r") as f:
        config_json = json.load(f)
    for stage, stage_config in config_json.items():
        print(f"{stage} params:")
        for param, value in stage_config.items():
            print(f"\t{param}: {value}")
    Path(config_json["data"]["task_dir"] + "/configs").mkdir(parents=True, exist_ok=True)
    config_file = config_json["data"]["task_dir"] + "/configs/" + args.config.split("/")[-1]
    if os.path.exists(config_file):
        print("config file already existed, use the existed one ...")
    else:
        copy2(args.config, config_file)


