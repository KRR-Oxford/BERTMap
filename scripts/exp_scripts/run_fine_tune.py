import os
main_dir = os.getcwd().split("BERTMap")[0] + "BERTMap"
import sys
sys.path.append(main_dir)
from bertmap.bert import OntoLabelBERT
from bertmap.utils import get_device, set_seed
from transformers import TrainingArguments
import torch
import json

torch.cuda.empty_cache()
# configurations
src = sys.argv[1]
tgt = sys.argv[2]
task_abbr = sys.argv[3]
assert task_abbr == "ss" or task_abbr == "us"
setting = sys.argv[4]
assert setting in ["f", "f+b", "f+b+i", "r", "f+r", "f+b+r", "f+b+i+r"]
data_base = f"{main_dir}/experiment/bert_fine_tune/data/{src}2{tgt}.{task_abbr}."
train_path = data_base + f"train.{setting}.tsv"
val_path = data_base + f"val.{setting}.tsv" if task_abbr == "us" else data_base + f"val.r.tsv"
test_path = data_base + f"test.r.tsv"
ckp_base = f"{main_dir}/experiment/bert_fine_tune/{src}2{tgt}.{task_abbr}.{setting}"
logging_steps = 100
eval_steps = 5 * logging_steps
train_epochs = 30  # for plain r setting, I think it should set to 20 epochs, but for others with big data, 10 epochs is enough.
batch_size = int(sys.argv[5])

training_args = TrainingArguments(
    output_dir=ckp_base,          
    num_train_epochs=train_epochs,              
    per_device_train_batch_size=batch_size,  # 32 or 16
    per_device_eval_batch_size=batch_size,   # 32 or 16
    warmup_steps=0,           
    weight_decay=0.01,   
    logging_steps=logging_steps,
    logging_dir=ckp_base+"/tb-logs",      
    eval_steps=eval_steps,
    evaluation_strategy="steps",
    do_train=True,
    do_eval=True,
    save_steps=eval_steps,
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=None,
    # disable_tqdm=True
)
set_seed(888)

# fine-tuning 
fine_tune = OntoLabelBERT("emilyalsentzer/Bio_ClinicalBERT", train_path, val_path, test_path, 
                          training_args, early_stop=True, huggingface=bool(int(sys.argv[6])))
fine_tune.trainer.train()
# evaluation on test set
test_results = fine_tune.trainer.evaluate(fine_tune.test)
test_results["train-val-test sizes"] = f"{len(fine_tune.train)}-{len(fine_tune.val)}-{len(fine_tune.test)}"
save_file = ckp_base + "/test.results.json"
with open(save_file, "w") as f:
    json.dump(test_results, f, indent=4, separators=(',', ': '), sort_keys=True)
