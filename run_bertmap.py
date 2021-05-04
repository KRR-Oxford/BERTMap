"""Script for running the whole process of BERTMap system as follows:
    1.
    2.
    3.
    4.

"""

# append the paths
import os

from scipy.sparse import construct
main_dir = os.getcwd().split("BERTMap")[0] + "BERTMap"
import sys
sys.path.append(main_dir)

# import essentials
import argparse
import json
import random
from shutil import copy2
from pathlib import Path

# import bertmap
from bertmap.utils import uniqify
from bertmap.onto import *
from bertmap.corpora import *

def fix_path(path_str: str):
    return main_dir + "/" + path_str

def banner(info=None, banner_len=60, sym="-"):
    print()
    if not info: 
        print(sym * banner_len)
    else: 
        info = sym * ((banner_len - len(info)) // 2 - 1) + " " + info
        info = info + " " + sym * (banner_len - len(info) - 1)
        print(info)
    print()
    
def prepare_data(config):
    
    # the task directory
    global task_dir, src, tgt, src_ob, tgt_ob, src_idx, tgt_idx
    
    # automatically set the task directory as {src}2{tgt} or {src}2{tgt}.{task}
    data_params = config["data"]
    src, tgt = data_params["src_onto"], data_params["tgt_onto"]
    task_dir = data_params["task_dir"]
    src_ob = None; tgt_ob = None
    
    # load the src ontology data files if already created
    if os.path.exists(task_dir + "/src.onto"): src_ob = OntoBox.from_saved(task_dir + "/src.onto")
    else: Path(task_dir + "/src.onto").mkdir(parents=True, exist_ok=True)
    # create the data files if not existed or missing
    if not src_ob: 
        src_ob = OntoBox(data_params["src_onto_file"], src, data_params["properties"], 
                                    config["bert"]["tokenizer_path"], data_params["cut"])
        src_ob.save(task_dir + "/src.onto")
    print(src_ob)
    
    # load the tgt ontology data files if already created
    if os.path.exists(task_dir + "/tgt.onto"): tgt_ob = OntoBox.from_saved(task_dir + "/tgt.onto")
    else: Path(task_dir + "/tgt.onto").mkdir(parents=True, exist_ok=True)
    # create the data files if not existed or missing
    if not tgt_ob: 
        tgt_ob = OntoBox(data_params["tgt_onto_file"], tgt, data_params["properties"], 
                                    config["bert"]["tokenizer_path"], data_params["cut"])
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
        ss_data = corpora.semi_supervised_data(**config["corpora"])
        banner("un-supervised data")
        us_data = corpora.unsupervised_data(**config["corpora"])
        Path(fine_tune_data_dir).mkdir(parents=True, exist_ok=True)
        with open(fine_tune_data_dir + "/ss.data.json", "w") as f: 
            json.dump(ss_data, f, indent=4, separators=(',', ': '), sort_keys=True)
        with open(fine_tune_data_dir + "/us.data.json", "w") as f: 
            json.dump(us_data, f, indent=4, separators=(',', ': '), sort_keys=True)
    
def fine_tune(config):
    pass

def compute_maps(config):
    pass

def eval_maps(config):
    pass

if __name__ == "__main__":
    
    # parse configuration file and specify mode
    parser = argparse.ArgumentParser(description='run bertmap system')
    parser.add_argument('-c', '--config', type=str, help='configuration file for bertmap system', required=True)
    parser.add_argument('-m', '--mode', type=str, choices={"pre", "train", "map", "all"}, default="all",
                        help='preprocessing data (pre), training BERT model (train), or computing the mappings and evaluate them (map)')
    args = parser.parse_args()
    
    banner("load configurations", sym="#")
    print(f"configuration-file: {args.config}")
    print(f"mode: {args.mode}")
    with open(args.config, "r") as f: 
        config = json.load(f)
    for stage, stage_config in config.items():
        print(f"{stage} params:")
        for param, value in stage_config.items():
            print(f"\t{param}: {value}")
    Path(config["data"]["task_dir"]).mkdir(parents=True, exist_ok=True)
    copy2(args.config, config["data"]["task_dir"])
            
    banner("prepare onto data", sym="#")
    prepare_data(config=config)
    
    banner("construct onto corpora and fine-tuning data", sym="#")
    construct_corpora(config=config)
    
    banner("trigger fine-tuning", sym="#")
    fine_tune(config=config)
    



