import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.bert import OntoLabelBERT
from bertmap.utils import get_device, set_seed
from transformers import TrainingArguments
import pandas as pd

# configurations
src = "fma"
tgt = "nci"
task = "semi-supervised"
task_abbr = "ss"
setting = "r"
data_base = f"/home/yuahe/projects/BERTMap/experiment/bert_fine_tune/data/{task}/{src}2{tgt}.{task_abbr}."
train_path = data_base + f"train.{setting}.tsv"
val_path = data_base + f"val.{setting}.tsv" if task == "unsupervised" else data_base + f"val.r.tsv"
test_path = data_base + f"test.r.tsv"
ckp_base = f"/home/yuahe/projects/BERTMap/experiment/bert_fine_tune/check_points/{task}/{src}2{tgt}.{task_abbr}.{setting}"
logging_steps = 100
eval_steps = 5 * logging_steps

training_args = TrainingArguments(
    output_dir=ckp_base,          
    num_train_epochs=50,              
    per_device_train_batch_size=64, 
    per_device_eval_batch_size=256,   
    warmup_steps=0,           
    weight_decay=0.01,   
    logging_steps=logging_steps,
    logging_dir=ckp_base+"/logs",      
    eval_steps=eval_steps,
    evaluation_strategy="steps",
    do_train=True,
    do_eval=True,
    save_steps=eval_steps,
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)
set_seed(888)

# fine-tuning 
fine_tune = OntoLabelBERT("emilyalsentzer/Bio_ClinicalBERT", train_path, val_path, test_path, training_args)
try:
    fine_tune.trainer.train()
    # evaluation on test set
    test_results = fine_tune.trainer.evaluate(fine_tune.test)
    pd.DataFrame.from_dict(test_results).to_csv(ckp_base/test_results.csv)
except:
    raise KeyboardInterrupt