# BERTMap

### Essential dependencies
The following packages are necessary but not sufficient for running BERTMap:
 ```
 conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch  # pytorch
 pip install cython  # the optimized parser of owlready2 relies on Cython
 pip install owlready2  # for managing ontologies
 pip install tensorboard  # tensorboard logging (optional)
 pip install transformers  # huggingface library
 pip install datasets  # huggingface datasets
 ```

### Running BERTMap
Clone the repository and run:
```
# fine-tuning
python run_bertmap.py -c config.json -m fine-tune 
# baseline
python run_bertmap.py -c config.json -m baseline
```
The script skips data construction once built for the first time to ensure that all of the models 
share the same set of pre-processed data. 

The fine-tuning model is implemented with huggingface Trainer, which by default uses multiple GPUs, 
for restricting to GPUs of specified indices, please run (for example):
```
# only device (1) and (2) are visible to the script
CUDA_VISIBLE_DEVICES=1,2 python run_bertmap.py -c config.json -m fine-tune 
```

### Configurations
Here gives the explanations of the variables used in `config.json`.
...
