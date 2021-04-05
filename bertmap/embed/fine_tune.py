"""
Pretrained BERT and its variants from Pytorch-based Huggingface Library.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch


class BinaryBERT:
    
    def __init__(self, pretrained_path):
        print("Initialize BERT for Binary Classification from the Pretrained BERT model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        
    