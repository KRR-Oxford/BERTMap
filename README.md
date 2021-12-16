# BERTMap: A BERT-based Ontology Alignment System

**Important Notices**
- The relevant paper was accepted in [AAAI-2022](https://aaai.org/Conferences/AAAI-22/).
- Arxiv version is available at: https://arxiv.org/abs/2112.02682.
- Code will be re-implemented as an example model in [DeepOnto](https://github.com/KRR-Oxford/DeepOnto), which will be a package for ontology engineering.


## About

BERTMap is a BERT-based ontology alignment system, which utilizes the textual knowledge of ontologies to fine-tune BERT and make prediction. It also incorporates sub-word inverted indices for candidate selection, and (graph-based) extension and (logic-based) repair modules for mapping refinement.

## Essential dependencies
The following packages are necessary but not sufficient for running BERTMap:
 ```
 conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch  # pytorch
 pip install cython  # the optimized parser of owlready2 relies on Cython
 pip install owlready2  # for managing ontologies
 pip install tensorboard  # tensorboard logging (optional)
 pip install transformers  # huggingface library
 pip install datasets  # huggingface datasets
 ```

## Running BERTMap

**IMPORTANT NOTICE**: BERTMap relies on class labels for training, but different ontologies have different annotation properties to define the aliases (synonyms), so preprocessing is required for adding all the synonyms to ``rdf:label`` before running BERTMap. The preprocessed ontologies involved in our paper together with their reference mappings are available in ``data.zip``.

Clone the repository and run:
```
# fine-tuning and evaluate bertmap prediction 
python run_bertmap.py -c config.json -m bertmap

# mapping extension (-e specify which mapping set {src, tgt, combined} to be extended)
python extend_bertmap.py -c config.json -e src

# evaluate extended bertmap 
python eval_bertmap.py -c config.json -e src

# repair and evluate final outputs (-t specify best validation threshold)
python repair_bertmap.py -c config.json -e src -t 0.999

# baseline models (edit similarity and pretrained bert embeddings)
python run_bertmap.py -c config.json -m nes
python run_bertmap.py -c config.json -m bertembeds
```
The script skips data construction once built for the first time to ensure that all of the models 
share the same set of pre-processed data. 

The fine-tuning model is implemented with huggingface Trainer, which by default uses multiple GPUs, 
for restricting to GPUs of specified indices, please run (for example):
```
# only device (1) and (2) are visible to the script
CUDA_VISIBLE_DEVICES=1,2 python run_bertmap.py -c config.json -m bertmap 
```

## Configurations
Here gives the explanations of the variables used in `config.json` for customized BERTMap running.

- `data`:
  - ``task_dir``: directory for saving all the output files.
  - ``src_onto``: source ontology name.
  - ``tgt_onto``: target ontology name.
  - ``task_suffix``: any suffix of the task if needed, e.g. the LargeBio track has 'small' and 'whole'.
  - ``src_onto_file``: source ontology file in ``.owl`` format.
  - ``tgt_onto_fil``: target ontology file in ``.owl`` format.
  - ``properties``: list of textual properties used for constructing semantic data , default is class labels: ``["label"]``.
  - ``cut``: threshold length for the ``keys`` of sub-word inverted index, preserve the ``keys`` only if their lengths > ``cut``, default is ``0``.
- `corpora`:
  - `sample_rate`: number of (soft) negative samples for each positive sample generated in corpora (not the ultimate fine-tuning data). 
  - `src2tgt_mappings_file`: reference mapping file for evaluation and semi-supervised learning setting in `.tsv` format with columns: ``"Entity1"``, ``"Entity2"`` and ``"Value"``.
  - ``ignored_mappings_file``: file in `.tsv` format but stores mappings that should be ignored by the evaluator.
  - `train_map_ratio`: proportion of training mappings to used in semi-supervised setting, default is ``0.2``.
  - `val_map_ratio`: proportion of validation mappings to used in semi-supervised setting, default is ``0.1``.
  - `test_map_ratio`: proportion of test mappings to used in semi-supervised setting, default is ``0.7``.
  - `io_soft_neg_rate`: number of soft negative sample for each positive sample generated in the fine-tuning data at the *intra-ontology* level.
  - `io_hard_neg_rate`: number of hard negative sample for each positive sample generated in the fine-tuning data at the *intra-ontology* level.
  - `co_soft_neg_rate`: number of soft negative sample for each positive sample generated in the fine-tuning data at the *cross-ontology* level.
  - `depth_threshold`: classes of depths larger than this threshold will not considered in hard negative generation, default is `null`.
  - `depth_strategy`: strategy to compute the depths of the classes if any threshold is set, default is `max`, choices are `max` and `min`.
- `bert`
  - `pretrained_path`: real or huggingface library path for pretrained BERT, e.g. `"emilyalsentzer/Bio_ClinicalBERT"` (BioClinicalBERT).
  - `tokenizer_path`: real or huggingface library path for BERT tokenizer, e.g. `"emilyalsentzer/Bio_ClinicalBERT"` (BioClinicalBERT).
- `fine-tune`
  - `include_ids`: include identity synonyms in the positive samples or not.
  - `learning`: choice of learning setting `ss` (semi-supervised) or `us` (unsupervised).
  - `warm_up_ratio`: portion of warm up steps.
  - `max_length`: maximum length for tokenizer (highly important for **large** task!).
  - `num_epochs`: number of training epochs, default is `3.0`.
  - `batch_size`: batch size for fine-tuning BERT.
  - `early_stop`: whether or not to apply early stopping (patience has been set to `10`), default is `false`.
  - `resume_checkpoint`: path to previous checkpoint if any, default is `null`.
- `map`
  - `candidate_limits`: list of candidate limits used for mapping computation, suggested values are `[25, 50, 100, 150, 200]`.
  - `batch_size`: batch size used for mapping computation.
  - `nbest`: number of top results to be considered.
  - `string_match`: whether or not to use string match before others.
  - `strategy`: strategy for classifier scoring method, default is `mean`.
- `eval`: 
  - `automatic`: whether or not automatically evaluate the mappings.

Should you need any further customizaions especially on the evaluation part, please set `eval: automatic` to `false` and use your own evaluation script.

## Acknolwedgements

The repair module is credited to [Ernesto Jiménez Ruiz et al.](http://www.cs.ox.ac.uk/isg/projects/LogMap/papers/paper_ISWC2011.pdf), and the code can be found [here](https://github.com/ernestojimenezruiz/logmap-matcher).

