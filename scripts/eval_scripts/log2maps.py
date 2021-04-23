import os
main_dir = os.getcwd().split("BERTMap")[0] + "BERTMap"
import sys
sys.path.append(main_dir)
from bertmap.map import OntoMapping

src = sys.argv[1]
tgt = sys.argv[2]
task = sys.argv[3]
setting = sys.argv[4]
candidate_limit = sys.argv[5]
log_dir = main_dir + f"/experiment/bert_fine_tune/{src}2{tgt}.{task}.{setting}/"
log = log_dir + f"map.{candidate_limit}.log"
OntoMapping.log2maps(log, keep=1)
