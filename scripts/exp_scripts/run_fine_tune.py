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
setting = "r"
data_base = f"/home/yuahe/projects/BERTMap/experiment/bert_fine_tune/data/{task}/{src}2{tgt}.ss."
train_path = data_base + f"train.{setting}.tsv"
val_path = data_base + f"val.{setting}.tsv"
test_path = data_base + f"test.{setting}.tsv"
ckp_base = f"/home/yuahe/projects/BERTMap/experiment/bert_fine_tune/check_points/{task}/{src}2{tgt}.ss.{setting}"
logging_steps = 100
eval_steps = 5 * logging_steps

training_args = TrainingArguments(
    output_dir=ckp_base,          # output directory
    max_steps=10000,              # total # of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=256,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_steps=logging_steps,
    logging_dir=ckp_base+"/logs",            # directory for storing logs
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
fine_tune.trainer.train()

# evaluation on test set
test_results = fine_tune.trainer.evaluate(fine_tune.test)
pd.DataFrame.from_dict(test_results).to_csv(ckp_base/test_results.csv)