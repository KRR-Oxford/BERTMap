"""Script for applying Mapping Extension algorithm on BERTMap results
"""

# append the paths
from bertmap.map.onto_map import OntoMapping
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
import subprocess

# import bertmap
from bertmap.utils import set_seed, banner
from bertmap.onto import OntoBox
from bertmap.extend import BERTClassifierExtend

na_vals = pd.io.parsers.STR_NA_VALUES.difference({"NULL", "null", "n/a"})


def mapping_repair(
    config,
    candidate_limit,
    set_type_to_repair,
    repair_threshold,
):

    # configurations
    global task_dir, exp_dir, map_dir, repaired_set_type, src_ob, tgt_ob
    task_dir = config["data"]["task_dir"]
    repaired_set_type = set_type_to_repair

    src_ob = OntoBox.from_saved(task_dir + "/src.onto")
    tgt_ob = OntoBox.from_saved(task_dir + "/tgt.onto")

    fine_tune_params = config["fine-tune"]
    learn = fine_tune_params["learning"]
    # assert learn == "us" or learn == "ss"
    include_ids = fine_tune_params["include_ids"]
    banner(f"repairing fine-tuned models of {learn} settings", sym="#")
    exp_dir = (
        task_dir + f"/fine-tune.exp/{learn}.exp"
        if not include_ids
        else task_dir + f"/fine-tune.exp/{learn}.ids.exp"
    )

    map_dir = f"{exp_dir}/map.{candidate_limit}"

    # apply mapping repair
    file_to_repair = f"{map_dir}/extended/{repaired_set_type}.{candidate_limit}.tsv"
    file_to_save_repaired = f"{map_dir}/repaired/{repaired_set_type}.{candidate_limit}.tsv"
    if os.path.exists(file_to_save_repaired):
        print(f"skip map repair for candidate limit {candidate_limit} as existed ...")
    else:
        Path(f"{map_dir}/repaired/").mkdir(parents=True, exist_ok=True)
        formatted_file_path = repair_formatting(file_to_repair, repair_threshold=repair_threshold)

        src_onto_path = task_dir + "/src.onto"
        for file in os.listdir(src_onto_path):
            if file.endswith(".owl"):
                src_onto_path += f"/{file}"
                break

        tgt_onto_path = task_dir + "/tgt.onto"
        for file in os.listdir(tgt_onto_path):
            if file.endswith(".owl"):
                tgt_onto_path += f"/{file}"
                break

        # apply java commands of LogMap DEBUGGER
        repair_command = (
            f"java -jar {main_dir}/repair_tools/logmap-matcher-4.0.jar DEBUGGER "
            + f"file:{src_onto_path} file:{tgt_onto_path} TXT {formatted_file_path} {map_dir}/repaired false true"
        )
        subprocess.run(repair_command.split(" "))
        eval_formatting(f"{map_dir}/repaired/mappings_repaired_with_LogMap.tsv", candidate_limit)
        
        
    # configure reference mappings and mappings to be ignored
    print(f"evaluate the repaired mappings with threshold: {repair_threshold}")
    pred = f"{map_dir}/repaired/{repaired_set_type}.{candidate_limit}.tsv"
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
    # train + val (30%) should be ignored for semi-supervised setting
    ref_ignored_ss = ref_ignored.append(val_maps_df).append(train_maps_df).reset_index(drop=True)
    # only val (10%) should be ignored for unsupervised setting
    ref_ignored_us = ref_ignored.append(val_maps_df).reset_index(drop=True)
    # results on 70% testing mappings
    result_ss = OntoMapping.evaluate(pred, ref, ref_ignored_ss, prefix=repaired_set_type)
    # results on 90% testing mappings
    result_us = OntoMapping.evaluate(pred, ref, ref_ignored_us, prefix=repaired_set_type)
    # save both results
    result_ss.to_csv(f"{map_dir}/repaired/results.test.ss.csv")
    result_us.to_csv(f"{map_dir}/repaired/results.test.us.csv")

def repair_formatting(map_file_tsv, repair_threshold):
    map_dict = BERTClassifierExtend.read_mappings_to_dict(map_file_tsv, threshold=repair_threshold)
    lines = []
    for m in map_dict.keys():
        src_iri, tgt_iri = m.split("\t")
        src_iri = src_ob.onto_text.expand_entity_iri(src_iri)
        tgt_iri = tgt_ob.onto_text.expand_entity_iri(tgt_iri)
        value = map_dict[m]
        lines.append(f"{src_iri}|{tgt_iri}|=|{value}|CLS\n")
    formatted_file = map_file_tsv.replace(".tsv", "-logmap_format.txt")
    with open(formatted_file, "w") as f:
        f.writelines(lines)
    return formatted_file


def eval_formatting(repaired_map_file_tsv, candidate_limit):
    repaired_df = pd.read_csv(
        repaired_map_file_tsv,
        sep="\t",
        names=["Entity1", "Entity2", "Value"],
        na_values=na_vals,
        keep_default_na=False,
    )
    repaired_df["Entity1"] = repaired_df["Entity1"].apply(
        lambda iri: src_ob.onto_text.abbr_entity_iri(iri)
    )
    repaired_df["Entity2"] = repaired_df["Entity2"].apply(
        lambda iri: tgt_ob.onto_text.abbr_entity_iri(iri)
    )
    repaired_df.to_csv(
        repaired_map_file_tsv.replace(
            "mappings_repaired_with_LogMap.tsv", f"{repaired_set_type}.{candidate_limit}.tsv"
        ),
        index=False,
        sep="\t",
    )


if __name__ == "__main__":

    set_seed(888)

    # parse configuration file and specify mode
    parser = argparse.ArgumentParser(description="run bertmap system")
    parser.add_argument(
        "-c", "--config", type=str, help="configuration file for bertmap system", required=True
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="threshold for mapping repair (suggested value is the best threshold from validation result)",
        default=0.999,
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
        print(f"current candidate limit: {limit}")
        mapping_repair(
            config=config_json,
            candidate_limit=limit,
            set_type_to_repair=args.extended,
            repair_threshold=args.threshold,
        )
        break
