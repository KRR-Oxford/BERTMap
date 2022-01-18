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
import multiprocessing_on_dill
import pandas as pd

# import bertmap
from bertmap import na_vals
from bertmap.utils import evenly_divide, set_seed, banner
from bertmap.map import OntoMapping

task_dir = ""
exp_dir = ""
map_dir = ""
extended_set_type = ""

def eval_maps(config, mode, candidate_limit, strategy=None, best_set_type=""):

    global task_dir, exp_dir, extended_set_type
    task_dir = config["data"]["task_dir"]
    extended_set_type = best_set_type

    if mode == "bertmap":
        fine_tune_params = config["fine-tune"]
        learn = fine_tune_params["learning"]
        # assert learn == "us" or learn == "ss"
        include_ids = fine_tune_params["include_ids"]
        banner(f"evaluate fine-tuned models of {learn} settings", sym="#")
        exp_dir = (
            task_dir + f"/fine-tune.exp/{learn}.exp"
            if not include_ids
            else task_dir + f"/fine-tune.exp/{learn}.ids.exp"
        )
        validate_then_test(config=config, candidate_limit=candidate_limit)
    elif mode == "bertembeds":
        if not strategy:
            for strt in ["cls", "mean"]:
                exp_dir = task_dir + f"/{strt}-embeds.exp"
                validate_then_test(config=config, candidate_limit=candidate_limit)
        else:
            exp_dir = task_dir + f"/{strategy}-embeds.exp"
            validate_then_test(config=config, candidate_limit=candidate_limit)
    elif mode == "edit":
        exp_dir = task_dir + "/nes.exp"
        validate_then_test(config=config, candidate_limit=candidate_limit)
    else:
        raise ValueError("invalid option of mode ...")
    
    return best_set_type


def validate_then_test(config, candidate_limit: int):

    global map_dir
    map_dir = f"{exp_dir}/map.{candidate_limit}"
    if extended_set_type:
        map_dir = f"{exp_dir}/map.{candidate_limit}/extended"

    best_bertmap_ind, best_strm_ind = validate_maps(config=config, candidate_limit=candidate_limit)
    if not best_bertmap_ind:
        # if already generated a validation results
        val_file = f"{map_dir}/results.val.{candidate_limit}.csv"
        val_results = pd.read_csv(val_file, index_col=0)
        best_bertmap_ind = list(val_results[:-3].idxmax()[["F1"]])[0]
        best_strm_ind = list(val_results[-3:].idxmax()[["F1"]])[0]
        banner(
            f"found best hyperparameters: {best_bertmap_ind} (BERTMap) {best_strm_ind} (String-match)"
        )
        # OntoMapping.print_eval(val_file, "(validation)")
    # generate 70% results for both unsupervised and semi-supervised setting for comparison
    test_maps(config, candidate_limit, best_bertmap_ind, best_strm_ind, semi_supervised=True)
    if "us" in str(config["fine-tune"]["learning"]):
        test_maps(config, candidate_limit, best_bertmap_ind, best_strm_ind, semi_supervised=False)


def test_maps(
    config, candidate_limit: int, best_hyper: str, best_strm_hyper: str, semi_supervised: bool
):

    if semi_supervised:
        eval_file = f"{map_dir}/results.test.ss.{candidate_limit}.csv"
    else:
        eval_file = f"{map_dir}/results.test.us.{candidate_limit}.csv"
    if os.path.exists(eval_file):
        print(f"skip map testing for candidate limit {candidate_limit} as existed ...")
        return

    # select the best mapping set-threshold combination according to validation results
    set_type, threshold = best_hyper.split(":")  # src/tgt/combined:threshold
    mapping_file = f"{map_dir}/{set_type}.{candidate_limit}.tsv"

    # configure reference mappings and mappings to be ignored
    ref = f"{task_dir}/refs/maps.ref.full.tsv"
    train_maps_df = pd.read_csv(
        f"{task_dir}/refs/maps.ref.ss.train.tsv", sep="\t", na_values=na_vals, keep_default_na=False
    )
    val_maps_df = pd.read_csv(
        f"{task_dir}/refs/maps.ref.ss.val.tsv", sep="\t", na_values=na_vals, keep_default_na=False
    )

    ref_ignored = (
        f"{task_dir}/refs/maps.ignored.tsv" if config["corpora"]["ignored_mappings_file"] else None
    )
    if ref_ignored:
        ref_ignored = pd.read_csv(ref_ignored, sep="\t", na_values=na_vals, keep_default_na=False)
    else:
        # init mappings to be ignored if there is no pre-defined one
        ref_ignored = pd.DataFrame(columns=["Entity1", "Entity2", "Value"])
    if semi_supervised:
        # train + val (30%) should be ignored for semi-supervised setting
        ref_ignored = ref_ignored.append(val_maps_df).append(train_maps_df).reset_index(drop=True)
    else:
        # only val (10%) should be ignored for unsupervised setting
        ref_ignored = ref_ignored.append(val_maps_df).reset_index(drop=True)

    # evaluate the corresponding test-set result
    result = OntoMapping.evaluate(mapping_file, ref, ref_ignored, float(threshold), set_type)
    # evaluate the baseline string-matching results
    set_type, threshold = best_strm_hyper.split(":")  # src/tgt/combined:threshold
    
    # This line fixes the bug on string-match evaluation
    mapping_file = f"{map_dir}/{set_type}.{candidate_limit}.tsv"
    
    result_strm = OntoMapping.evaluate(mapping_file, ref, ref_ignored, float(threshold), set_type)
    result = result.append(result_strm)

    result.to_csv(eval_file)
    if semi_supervised:
        banner("70% test set results (semi-supervised)")
    else:
        banner("90% test set results (unsupervised)")
    print(result)
    return result


def validate_maps(config, candidate_limit: int):

    eval_file = f"{map_dir}/results.val.{candidate_limit}.csv"
    if os.path.exists(eval_file):
        print(f"skip map validation for candidate limit {candidate_limit} as existed ...")
        return None, None

    report = pd.DataFrame(columns=["#Mappings", "#Ignored", "Precision", "Recall", "F1"])
    ref = f"{task_dir}/refs/maps.ref.full.tsv"
    ref_ignored = (
        f"{task_dir}/refs/maps.ignored.tsv" if config["corpora"]["ignored_mappings_file"] else None
    )
    if ref_ignored:
        ref_ignored = pd.read_csv(ref_ignored, sep="\t", na_values=na_vals, keep_default_na=False)
    else:
        # init mappings to be ignored if there is no pre-defined one
        ref_ignored = pd.DataFrame(columns=["Entity1", "Entity2", "Value"])
    train_maps_df = pd.read_csv(
        f"{task_dir}/refs/maps.ref.ss.train.tsv", sep="\t", na_values=na_vals, keep_default_na=False
    )
    test_maps_df = pd.read_csv(
        f"{task_dir}/refs/maps.ref.ss.test.tsv", sep="\t", na_values=na_vals, keep_default_na=False
    )
    # during validation, training and testing mappings should be ignored
    ref_ignored = ref_ignored.append(train_maps_df).append(test_maps_df).reset_index(drop=True)

    pool = multiprocessing_on_dill.Pool(10)
    eval_results = []
    thresholds = (
        evenly_divide(0, 0.8, 8) + evenly_divide(0.9, 0.97, 7) + evenly_divide(0.98, 1.0, 20)
    )
    
    cb_map_path = f"{exp_dir}/map.{candidate_limit}/combined.{candidate_limit}.tsv"
    src_map_path = f"{exp_dir}/map.{candidate_limit}/src.{candidate_limit}.tsv"
    tgt_map_path = f"{exp_dir}/map.{candidate_limit}/tgt.{candidate_limit}.tsv"
    
    if extended_set_type == "combined":
        cb_map_path = f"{map_dir}/combined.{candidate_limit}.tsv"
    elif extended_set_type == "src":
        src_map_path = f"{map_dir}/src.{candidate_limit}.tsv"
    elif extended_set_type == "tgt":
        tgt_map_path = f"{map_dir}/tgt.{candidate_limit}.tsv"
    
    for threshold in thresholds:
        threshold = round(threshold, 6)
        eval_results.append(
            pool.apply_async(
                OntoMapping.evaluate,
                args=(
                    cb_map_path,
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
                    src_map_path,
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
                    tgt_map_path,
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
    OntoMapping.print_eval(eval_file, "(validation)")

    # return the best validation hyperparameter
    best_bertmap_ind = list(report[:-3].idxmax()[["F1"]])[0]
    best_string_match_ind = list(report[-3:].idxmax()[["F1"]])[0]
    return best_bertmap_ind, best_string_match_ind


if __name__ == "__main__":

    set_seed(888)

    # parse configuration file and specify mode
    parser = argparse.ArgumentParser(description="run bertmap system")
    parser.add_argument(
        "-c", "--config", type=str, help="configuration file for bertmap system", required=True
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices={"bertmap", "bertembeds", "edit"},
        default="bertmap",
        help="preprocessing data (pre), training BERT model (train), or computing the mappings and evaluate them (map)",
    )
    parser.add_argument(
        "-e",
        "--extended",
        type=str,
        choices={"src", "tgt", "combined", ""},
        default="",
        help="the best set type from first round of validation",
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

    for limit in config_json["map"]["candidate_limits"]:
        eval_maps(config=config_json, mode=args.mode, candidate_limit=limit, best_set_type=args.extended)
