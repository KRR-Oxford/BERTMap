"""Script for running the whole process of BERTMap system as follows:

    1. parse source and target ontologies and retrieve classtexts and sub-word index 
    2. construct corpora for both un-supervised and semi-supervised settings 
    3. fine-tuning BERT classifier if "bertmap" chosen for mode
    4. compute mappings for selected model including
        (a) "bertmap": BERT fine-tuned classifier;
        (b) "bertembeds": BERT embeddings + cosine-similarity;
        (c) "edit": normalized-edit-distance.
    5. evaluate the computed mappings (one can disable the evaluation part and insert your customized evaluation method)

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


def fix_path(path_str: str):
    return main_dir + "/" + path_str


def prepare_data(config):

    # the task directory
    global task_dir, src, tgt, src_ob, tgt_ob

    # automatically set the task directory as {src}2{tgt} or {src}2{tgt}.{task}
    data_params = config["data"]
    src, tgt = data_params["src_onto"], data_params["tgt_onto"]
    task_dir = data_params["task_dir"]
    src_ob = None
    tgt_ob = None

    # load the src ontology data files if already created
    if os.path.exists(task_dir + "/src.onto"):
        src_ob = OntoBox.from_saved(task_dir + "/src.onto")
    else:
        Path(task_dir + "/src.onto").mkdir(parents=True, exist_ok=True)
    # create the data files if not existed or missing
    if not src_ob:
        src_ob = OntoBox(
            data_params["src_onto_file"],
            src,
            data_params["properties"],
            config["bert"]["tokenizer_path"],
            data_params["cut"],
        )
        src_ob.save(task_dir + "/src.onto")
    print(src_ob)

    # load the tgt ontology data files if already created
    if os.path.exists(task_dir + "/tgt.onto"):
        tgt_ob = OntoBox.from_saved(task_dir + "/tgt.onto")
    else:
        Path(task_dir + "/tgt.onto").mkdir(parents=True, exist_ok=True)
    # create the data files if not existed or missing
    if not tgt_ob:
        tgt_ob = OntoBox(
            data_params["tgt_onto_file"],
            tgt,
            data_params["properties"],
            config["bert"]["tokenizer_path"],
            data_params["cut"],
        )
        tgt_ob.save(task_dir + "/tgt.onto")
    print(tgt_ob)


def construct_corpora(config):
    global corpora, fine_tune_data_dir
    if os.path.exists(task_dir + "/refs") and os.path.exists(task_dir + "/corpora"):
        corpora = OntoAlignCorpora.from_saved(task_dir)
    else:
        corpora = OntoAlignCorpora(src_ob=src_ob, tgt_ob=tgt_ob, **config["corpora"], from_saved=False)
        corpora.save(task_dir)
    print(corpora)
    # Note: it is recommended to set co_soft_neg_rate = io_soft_neg_rate + io_hard_neg_rate
    # to ensure similar proportion of negative samples at both intra-onto and cross-onto level
    fine_tune_data_dir = task_dir + "/fine-tune.data"
    if not os.path.exists(fine_tune_data_dir):
        banner("semi-supervised data")
        ss_data, ss_report = corpora.semi_supervised_data(**config["corpora"])
        banner("un-supervised data")
        us_data, us_report = corpora.unsupervised_data(**config["corpora"])
        Path(fine_tune_data_dir).mkdir(parents=True, exist_ok=True)
        with open(fine_tune_data_dir + "/ss.data.json", "w") as f:
            json.dump(ss_data, f, indent=4, separators=(",", ": "), sort_keys=True)
        with open(fine_tune_data_dir + "/ss.info", "w") as f:
            f.write(ss_report)
        with open(fine_tune_data_dir + "/us.data.json", "w") as f:
            json.dump(us_data, f, indent=4, separators=(",", ": "), sort_keys=True)
        with open(fine_tune_data_dir + "/us.info", "w") as f:
            f.write(us_report)


def fine_tune(config):

    global exp_dir, learn

    fine_tune_params = config["fine-tune"]
    learn = fine_tune_params["learning"]
    assert learn == "us" or learn == "ss"
    include_ids = fine_tune_params["include_ids"]
    data_file = fine_tune_data_dir + f"/{learn}.data.json"
    banner(f"fine-tuning on {learn} settings", sym="#")
    exp_dir = (
        task_dir + f"/fine-tune.exp/{learn}.exp" if not include_ids else task_dir + f"/fine-tune.exp/{learn}.ids.exp"
    )
    if os.path.exists(exp_dir) and fine_tune_params["resume_checkpoint"] is None:
        banner(f"skip fine-tuning as checkpoints exist")
        return

    with open(data_file, "r") as file:
        oa_data = json.load(file)
    train = oa_data["train+"] if include_ids else oa_data["train"]
    val = oa_data["val+"] if learn == "us" and include_ids else oa_data["val"]
    test = oa_data["test"]

    torch.cuda.empty_cache()
    bert_oa = BERTTrainer(
        config["bert"]["pretrained_path"],
        train,
        val,
        test,
        max_length=fine_tune_params["max_length"],
        early_stop=fine_tune_params["early_stop"],
        early_stop_patience=10,
    )

    batch_size = fine_tune_params["batch_size"]
    epoch_steps = len(bert_oa.tra) // batch_size  # total steps of an epoch
    # keep logging steps consisitent even for small batch size
    # report logging on every 0.02 epoch
    logging_steps = int(epoch_steps * 0.02)
    # eval on every 0.1 epoch
    eval_steps = 5 * logging_steps

    training_args = TrainingArguments(
        output_dir=exp_dir,
        # max_steps=eval_steps*4 + 1,
        num_train_epochs=fine_tune_params["num_epochs"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=fine_tune_params["warm_up_ratio"],
        weight_decay=0.01,
        logging_steps=logging_steps,
        logging_dir=f"{exp_dir}/tb",
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        do_train=True,
        do_eval=True,
        save_steps=eval_steps,
        load_best_model_at_end=True,
        save_total_limit=1,
        # metric_for_best_model="accuracy",
        # greater_is_better=True,
    )

    bert_oa.train(training_args, resume_from_ckp=fine_tune_params["resume_checkpoint"])
    # evaluation on test set
    test_results = bert_oa.trainer.evaluate(bert_oa.tst)
    test_results["train-val-test sizes"] = f"{len(bert_oa.tra)}-{len(bert_oa.val)}-{len(bert_oa.tst)}"
    test_results_file = exp_dir + "/test.results.json"
    with open(test_results_file, "w") as file:
        json.dump(test_results, file, indent=4, separators=(",", ": "), sort_keys=True)


def compute_fine_tune_maps(config):

    global checkpoint
    checkpoint = exp_dir
    for file in os.listdir(exp_dir):
        if file.startswith("checkpoint"):
            checkpoint += f"/{file}"
            break
    best_ckp = checkpoint.split("/")[-1]
    banner(f"find best {best_ckp}")

    map_params = deepcopy(config["map"])
    limits = map_params["candidate_limits"]
    del map_params["candidate_limits"]
    for candidate_limit in limits:
        map_file = f"{exp_dir}/map.{candidate_limit}/map.{candidate_limit}.log"
        if os.path.exists(map_file):
            print(f"skip map computation for candidate limit {candidate_limit} as existed ...")
        else:
            Path(f"{exp_dir}/map.{candidate_limit}").mkdir(parents=True, exist_ok=True)
            mapping_computer = BERTClassifierMapping(
                src_ob=src_ob,
                tgt_ob=tgt_ob,
                candidate_limit=candidate_limit,
                bert_checkpoint=checkpoint,
                tokenizer_path=config["bert"]["tokenizer_path"],
                save_dir=f"{exp_dir}/map.{candidate_limit}",
                max_length=config["fine-tune"]["max_length"],
                **map_params,
            )
            mapping_computer.run()
            src_df, tgt_df, combined_df = OntoMapping.read_mappings_from_log(
                f"{exp_dir}/map.{candidate_limit}/map.{candidate_limit}.log", keep=1
            )
            src_df.to_csv(f"{exp_dir}/map.{candidate_limit}/src.{candidate_limit}.tsv", sep="\t", index=False)
            tgt_df.to_csv(f"{exp_dir}/map.{candidate_limit}/tgt.{candidate_limit}.tsv", sep="\t", index=False)
            combined_df.to_csv(
                f"{exp_dir}/map.{candidate_limit}/combined.{candidate_limit}.tsv", sep="\t", index=False
            )
            banner(f"evaluate mappings for candidate limit {candidate_limit}")
            time.sleep(10)
            torch.cuda.empty_cache()
        if config["eval"]["automatic"]:
            eval_maps(config=config, candidate_limit=candidate_limit)
            if learn == "ss":
                eval_maps(config=config, candidate_limit=candidate_limit, semi_supervised=True)


def compute_embeds_maps(config):

    map_params = deepcopy(config["map"])
    limits = map_params["candidate_limits"]
    del map_params["candidate_limits"]
    del map_params["strategy"]

    for strategy in ["mean", "cls"]:
        global exp_dir
        exp_dir = task_dir + f"/{strategy}-embeds.exp"
        if os.path.exists(exp_dir):
            print("found /embeds.exp directory")
        else:
            Path(f"{exp_dir}").mkdir(parents=True, exist_ok=True)

        for candidate_limit in limits:
            map_file = f"{exp_dir}/map.{candidate_limit}/map.{candidate_limit}.log"
            if os.path.exists(map_file):
                print(f"skip map computation for candidate limit {candidate_limit} as existed ...")
            else:
                Path(f"{exp_dir}/map.{candidate_limit}").mkdir(parents=True, exist_ok=True)

                mapping_computer = BERTEmbedsMapping(
                    src_ob=src_ob,
                    tgt_ob=tgt_ob,
                    candidate_limit=candidate_limit,
                    bert_checkpoint=config["bert"]["pretrained_path"],
                    tokenizer_path=config["bert"]["tokenizer_path"],
                    save_dir=f"{exp_dir}/map.{candidate_limit}",
                    strategy=strategy,
                    max_length=config["fine-tune"]["max_length"],
                    **map_params,
                )
                mapping_computer.run()
                src_df, tgt_df, combined_df = OntoMapping.read_mappings_from_log(
                    f"{exp_dir}/map.{candidate_limit}/map.{candidate_limit}.log", keep=1
                )
                src_df.to_csv(f"{exp_dir}/map.{candidate_limit}/src.{candidate_limit}.tsv", sep="\t", index=False)
                tgt_df.to_csv(f"{exp_dir}/map.{candidate_limit}/tgt.{candidate_limit}.tsv", sep="\t", index=False)
                combined_df.to_csv(
                    f"{exp_dir}/map.{candidate_limit}/combined.{candidate_limit}.tsv", sep="\t", index=False
                )
                banner(f"evaluate mappings for candidate limit {candidate_limit}")
                banner(f"evaluate mappings for candidate limit {candidate_limit}")
                time.sleep(10)
                torch.cuda.empty_cache()
            if config["eval"]["automatic"]:
                eval_maps(config=config, candidate_limit=candidate_limit)
                eval_maps(config=config, candidate_limit=candidate_limit, semi_supervised=True)


def compute_nes_maps(config):

    global exp_dir
    exp_dir = task_dir + "/nes.exp"
    if os.path.exists(exp_dir):
        print("found /nes.exp directory")
    else:
        Path(f"{exp_dir}").mkdir(parents=True, exist_ok=True)

    map_params = deepcopy(config["map"])
    limits = map_params["candidate_limits"]

    for candidate_limit in limits:
        map_file = f"{exp_dir}/map.{candidate_limit}/map.{candidate_limit}.log"
        if os.path.exists(map_file):
            print(f"skip map computation for candidate limit {candidate_limit} as existed ...")
        else:
            Path(f"{exp_dir}/map.{candidate_limit}").mkdir(parents=True, exist_ok=True)
            mapping_computer = NormEditSimMapping(
                src_ob=src_ob,
                tgt_ob=tgt_ob,
                candidate_limit=candidate_limit,
                save_dir=f"{exp_dir}/map.{candidate_limit}",
            )
            # mapping_computer.run()  # single-thread NES experiment
            ############## chunk for setting up multiprocessing on NES experiment ############
            mapping_computer.start_time = time.time()
            procs = []
            src_classes = list(src_ob.onto.classes())
            src_idxs = equal_split(8, len(src_classes))
            tgt_classes = list(tgt_ob.onto.classes())
            tgt_idxs = equal_split(8, len(tgt_classes))

            def align_batch(batch_classes, flag):
                for cl in batch_classes:
                    mapping_computer.align_one_class(cl, flag=flag)

            for idxs in src_idxs:
                batch = [src_classes[i] for i in idxs]
                p = multiprocessing_on_dill.Process(
                    target=align_batch,
                    args=(
                        batch,
                        "SRC",
                    ),
                )
                p.start()
                procs.append(p)
            for idxs in tgt_idxs:
                batch = [tgt_classes[i] for i in idxs]
                p = multiprocessing_on_dill.Process(
                    target=align_batch,
                    args=(
                        batch,
                        "TGT",
                    ),
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            ############## chunk for setting up multiprocessing on NES experiment ############

            src_df, tgt_df, combined_df = OntoMapping.read_mappings_from_log(
                f"{exp_dir}/map.{candidate_limit}/map.{candidate_limit}.log", keep=1
            )
            src_df.to_csv(f"{exp_dir}/map.{candidate_limit}/src.{candidate_limit}.tsv", sep="\t", index=False)
            tgt_df.to_csv(f"{exp_dir}/map.{candidate_limit}/tgt.{candidate_limit}.tsv", sep="\t", index=False)
            combined_df.to_csv(
                f"{exp_dir}/map.{candidate_limit}/combined.{candidate_limit}.tsv", sep="\t", index=False
            )
            banner(f"evaluate mappings for candidate limit {candidate_limit}")
            time.sleep(10)
        if config["eval"]["automatic"]:
            eval_maps(config=config, candidate_limit=candidate_limit)
            eval_maps(config=config, candidate_limit=candidate_limit, semi_supervised=True)


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

    banner("prepare onto data", sym="#")
    prepare_data(config=config_json)

    banner("construct onto corpora and fine-tuning data", sym="#")
    construct_corpora(config=config_json)

    if args.mode == "bertmap":
        fine_tune(config=config_json)
        banner("compute and evaluate fine-tuning mappings", sym="#")
        compute_fine_tune_maps(config=config_json)
    elif args.mode == "bertembeds":
        banner(f"compute and evaluate baseline BERT embeddings mappings", sym="#")
        compute_embeds_maps(config=config_json)
    elif args.mode == "edit":
        banner(f"compute and evaluate normalized edit distance mappings", sym="#")
        compute_nes_maps(config=config_json)
