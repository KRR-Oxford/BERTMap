main_dir = "/home/yuahe/projects/BERTMap"
import sys
sys.path.append(main_dir)
from bertmap.map import OntoMapping

src = "fma"
tgt = "nci"
task = "us"
setting = "f"
log_dir = main_dir + f"/experiment/bert_fine_tune/{src}2{tgt}.{task}.{setting}/"
log = log_dir + "select.log"
OntoMapping.logs2maps(log, keep=1)
