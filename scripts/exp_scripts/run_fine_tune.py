main_dir = "C:/Users/lawhy/Work/Oxford-SRUK-OntoAlign/BERTMap"
import sys
sys.path.append(main_dir)
from bertmap.bert import OntoLabelBERT
from bertmap.utils import get_device, set_seed
from transformers import TrainingArguments
import json

task_dict = {"ss": "semi-supervised", "us": "unsupervised"}

# configurations
src = "fma"
tgt = "nci"
task_abbr = sys.argv[1]
assert task_abbr == "ss" or task_abbr == "us"
task = task_dict[task_abbr]
setting = sys.argv[2]
assert setting in ["f", "f+b", "f+b+i", "r", "f+r", "f+b+r", "f+b+i+r"]
data_base = f"{main_dir}/experiment/bert_fine_tune/data/{task}/{src}2{tgt}.{task_abbr}."
train_path = data_base + f"train.{setting}.tsv"
val_path = data_base + f"val.{setting}.tsv" if task == "unsupervised" else data_base + f"val.r.tsv"
test_path = data_base + f"test.r.tsv"
ckp_base = f"{main_dir}/experiment/bert_fine_tune/check_points/{task}/{src}2{tgt}.{task_abbr}.{setting}"
logging_steps = 200
eval_steps = 5 * logging_steps
train_epochs = int(sys.argv[3])  # for plain r setting, I think it should set to 20 epochs, but for others with big data, 10 epochs is enough.

training_args = TrainingArguments(
    output_dir=ckp_base,          
    num_train_epochs=train_epochs,              
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=512,   
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
    # disable_tqdm=True
)
set_seed(888)

# with open(f"{ckp_base}/train.log", "w") as log:
#     with redirect_stdout(log):
# fine-tuning 
fine_tune = OntoLabelBERT("emilyalsentzer/Bio_ClinicalBERT", train_path, val_path, test_path, training_args, early_stop=True)
fine_tune.trainer.train()
# evaluation on test set
test_results = fine_tune.trainer.evaluate(fine_tune.test)
test_results["train-val-test sizes"] = f"{len(fine_tune.train)}-{len(fine_tune.val)}-{len(fine_tune.test)}"
save_file = ckp_base + "/test.results.json"
with open(save_file, "w") as f:
    json.dump(test_results, f, indent=4, separators=(',', ': '), sort_keys=True)
