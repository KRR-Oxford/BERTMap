"""Script for applying Mapping Extension algorithm on BERTMap results
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
import pandas as pd

# import bertmap
from bertmap.utils import set_seed, banner
from bertmap.onto import OntoBox
from bertmap.extend import BERTClassifierExtend

na_vals = pd.io.parsers.STR_NA_VALUES.difference({"NULL", "null", "n/a"})


def mapping_extension(config, candidate_limit, set_type_to_extend="", mapping_threshold=0.900):

    global task_dir, exp_dir, map_dir, extended_set_type, src_ob, tgt_ob, theta
    task_dir = config["data"]["task_dir"]
    extended_set_type = set_type_to_extend
    theta = mapping_threshold

    src_ob = OntoBox.from_saved(task_dir + "/src.onto")
    tgt_ob = OntoBox.from_saved(task_dir + "/tgt.onto")

    fine_tune_params = config["fine-tune"]
    learn = fine_tune_params["learning"]
    # assert learn == "us" or learn == "ss"
    include_ids = fine_tune_params["include_ids"]
    banner(f"extending fine-tuned models of {learn} settings", sym="#")
    exp_dir = (
        task_dir + f"/fine-tune.exp/{learn}.exp"
        if not include_ids
        else task_dir + f"/fine-tune.exp/{learn}.ids.exp"
    )

    map_dir = f"{exp_dir}/map.{candidate_limit}"
    file_to_extend = f"{map_dir}/{extended_set_type}.{candidate_limit}.tsv"
    file_to_save = f"{map_dir}/extended/{extended_set_type}.{candidate_limit}.tsv"
    if os.path.exists(file_to_save):
        print(f"skip map extension for candidate limit {candidate_limit} as existed ...")
        return
    else:
        Path(f"{map_dir}/extended/").mkdir(parents=True, exist_ok=True)
    
    banner(f"apply mapping extension on {extended_set_type} mappings")

    # load fine-tuned BERT classifier
    checkpoint = exp_dir
    for file in os.listdir(exp_dir):
        if file.startswith("checkpoint"):
            checkpoint += f"/{file}"
            break
    best_ckp = checkpoint.split("/")[-1]
    print(f"find best {best_ckp}")

    bert_ex = BERTClassifierExtend(
        src_ob=src_ob,
        tgt_ob=tgt_ob,
        mapping_file=file_to_extend,
        extend_threshold=theta,
        bert_checkpoint=checkpoint,
        tokenizer_path=config["bert"]["tokenizer_path"],
        max_length=config["fine-tune"]["max_length"],
    )
    
    bert_ex.extend_mappings(max_iter=50)
    
    exp_list = []
    for m, v in bert_ex.expansion.items():
        src_iri, tgt_iri = m.split("\t")
        exp_list.append((src_iri, tgt_iri, v))
    exp_df = pd.DataFrame(exp_list, columns=["Entity1", "Entity2", "Value"])
        
    pred_df = pd.read_csv(file_to_extend, sep="\t", na_values=na_vals, keep_default_na=False)
    extended_pred_df = pred_df.append(exp_df).reset_index(drop=True)
    extended_pred_df.to_csv(file_to_save, index=False, sep="\t")
    
    print(f"# mappings: before={len(pred_df)} after={len(extended_pred_df)}")


if __name__ == "__main__":

    set_seed(888)

    # parse configuration file and specify mode
    parser = argparse.ArgumentParser(description="run bertmap system")
    parser.add_argument(
        "-c", "--config", type=str, help="configuration file for bertmap system", required=True
    )
    parser.add_argument(
        "-t", "--threshold", type=float, help="threshold for mapping extension", default=0.900
    )
    parser.add_argument(
        "-e",
        "--extended",
        type=str,
        choices={"src", "tgt", "combined"},
        required=True,
        help="the mapping set type to be extended",
    )
    args = parser.parse_args()

    banner("load configurations", sym="#")
    print(f"configuration-file: {args.config}")
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
        mapping_extension(
            config=config_json,
            candidate_limit=limit,
            set_type_to_extend=args.extended,
            mapping_threshold=args.threshold,
        )
