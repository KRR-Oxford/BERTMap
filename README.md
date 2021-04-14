# BERTMap

### Code and Data Management

Data and Experiment files are available at [here](https://drive.google.com/drive/folders/11_Dj6f7MN3pTKkWUUAKnY-vTWh4mMOdE?usp=sharing).

Please clone this repo and download the folders ``data/`` and ``experiment/`` from the link into the BERTMap repository such that:
```
BERTMap/

  bertmap/  # essential code
  data/  # storing different types of data, e.g. .owl (ontologies); .tsv (for class2text data and mappings); .json (for corpora)
  experiment/  # storing different experiment settings and their results
  scripts/  # scripts for generating data, running experiments and evaluating the computed mappings
  .gitignore
  README.md
  
```

### Scripts Usage

#### Data Preparation
- *Generate the Class2Text (labels) data file*: ``data_scripts/save_class2text.py``.
-------------------
#### Baseline
- *BERT Baseline Experiment*: ``data_scripts/save_bert_class_embeds.py`` ➡️ ``exp_scripts/run_bert.py`` ➡️ ``eval_scripts/eval_bert.py``.
- *Normalized Edit Distance Baseline Experiment*: ``exp_scripts/run_nes.py`` ➡️ ``eval_scripts/eval_nes.py``.  (Not recommended to run on the large ontologies because its time complextity is `O(n^2)`).
--------------------
#### Fine-tuning on synonym/nonsynonym corpora
- *Generate the synonym/nonsynonym corpora*: ``corpora_scripts/save_*_corpora.py`.
- *Sampling training/dev/test sets from corpora for different experiment settings*: ``exp_scripts/set_fine_tune_data.py``
- *Fine-tuning BERT and evaluate on intermediate test set*: ``exp_scripts/run_fine_tune.py``
- *Mapping Prediction using Fine-tuned BERT*: ...
