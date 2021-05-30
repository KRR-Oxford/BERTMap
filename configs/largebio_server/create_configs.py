import json
from pathlib import Path


server_dir = "/home/cjy/BERTMap"
cur_dir = "/home/yuahe/projects/BERTMap"

with open(f"{cur_dir}/config.json", "r") as f:
    config = json.load(f)

for src, tgt in [
    ("fma", "nci"),
    ("fma", "snomed"),
    ("snomed", "nci"),
    ("fma", "snomed+"),
    ("snomed+", "nci"),
]:
    for learn in ["us", "ss"]:
        for ids in [True, False]:
            exp_dir = f"{cur_dir}/configs/largebio/{src}2{tgt}.small"
            Path(exp_dir).mkdir(parents=True, exist_ok=True)
            config["data"]["task_dir"] = server_dir + f"/{src}2{tgt}.small"
            config["data"]["src_onto"] = src
            config["data"]["tgt_onto"] = tgt
            config["data"]["src_onto_file"] = (
                server_dir + f"/data/largebio/ontos/{src}2{tgt}.small.owl"
            )
            config["data"]["tgt_onto_file"] = (
                server_dir + f"/data/largebio/ontos/{tgt}2{src}.small.owl"
            )
            src_ori = src.replace("+", "")
            tgt_ori = tgt.replace("+", "")
            config["corpora"]["src2tgt_mappings_file"] = (
                server_dir + f"/data/largebio/refs/{src_ori}2{tgt_ori}.tsv"
            )
            config["corpora"]["ignored_mappings_file"] = (
                server_dir + f"/data/largebio/refs/{src_ori}2{tgt_ori}.ignored.tsv"
            )
            config["fine-tune"]["include_ids"] = ids
            config["fine-tune"]["learning"] = learn
            config_file = (
                exp_dir + f"/{learn}.ids.config.json" if ids else exp_dir + f"/{learn}.config.json"
            )
            config["map"]["batch_size"] = 72
            with open(config_file, "w") as c:
                json.dump(config, c, indent=4, separators=(",", ": "))


for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
    for learn in ["us", "ss"]:
        for ids in [True, False]:
            exp_dir = f"{cur_dir}/configs/largebio/{src}2{tgt}.whole"
            Path(exp_dir).mkdir(parents=True, exist_ok=True)
            config["data"]["task_dir"] = server_dir + f"/{src}2{tgt}.whole"
            config["data"]["src_onto"] = src
            config["data"]["tgt_onto"] = tgt
            config["data"]["src_onto_file"] = server_dir + f"/data/largebio/ontos/{src}.whole.owl"
            config["data"]["tgt_onto_file"] = server_dir + f"/data/largebio/ontos/{tgt}.whole.owl"
            src_ori = src.replace("+", "")
            tgt_ori = tgt.replace("+", "")
            config["corpora"]["src2tgt_mappings_file"] = (
                server_dir + f"/data/largebio/refs/{src_ori}2{tgt_ori}.tsv"
            )
            config["corpora"]["ignored_mappings_file"] = (
                server_dir + f"/data/largebio/refs/{src_ori}2{tgt_ori}.ignored.tsv"
            )
            config["fine-tune"]["include_ids"] = ids
            config["fine-tune"]["learning"] = learn
            config_file = (
                exp_dir + f"/{learn}.ids.config.json" if ids else exp_dir + f"/{learn}.config.json"
            )
            config["map"]["batch_size"] = 32
            with open(config_file, "w") as c:
                json.dump(config, c, indent=4, separators=(",", ": "))
