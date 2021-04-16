from bertmap.map.direct_search import DirectSearchMapping
from bertmap.bert import PretrainedBERT
from bertmap.onto import Ontology
import torch

class DirectBERTClassifierMapping(DirectSearchMapping):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_class2text_tsv, tgt_onto_class2text_tsv, save_path, 
                 batch_size, nbest=2, task_suffix="small", name="bc-tuned-mean", 
                 bert_path="", tokenizer_path="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_class2text_tsv, 
                         tgt_onto_class2text_tsv, save_path, task_suffix=task_suffix, name=name)    
        self.batch_size = batch_size   
        self.nbest = nbest
        self.bert = PretrainedBERT(pretrained_path=bert_path, tokenizer_path=tokenizer_path, with_classifier=True)
        self.softmax = torch.nn.Softmax(dim=1)
        # pass the binary classification scores into a softmax layer and choose the synonym score (column 1, y=1)
        self.classifier = lambda x: self.softmax(self.bert.model(**self.bert.tokenizer(x, padding=True, return_tensors="pt")).logits)[:, 1]
        self.strategy = name.split("-")[2]  # ["bc", "tuned", "mean"]
        
    def align_config(self, flag="SRC"):
        assert flag == "SRC" or flag == "TGT"
        from_onto_class2text_path = self.src_onto_class2text_path
        to_onto_class2text_path = self.tgt_onto_class2text_path
        mappings = self.src2tgt_mappings
        if flag == "TGT":
            from_onto_class2text_path, to_onto_class2text_path = to_onto_class2text_path, from_onto_class2text_path
            mappings = self.tgt2src_mappings
        return from_onto_class2text_path, to_onto_class2text_path, mappings
    
    def fixed_one_side_alignment(self, flag="SRC"):
        from_onto_class2text_path, to_onto_class2text_path, mappings = self.align_config(flag=flag)
        from_onto_class2text = Ontology.load_class2text(from_onto_class2text_path)
        for i, dp in from_onto_class2text.iterrows():
            to_batch_generator = Ontology.class2text_batch_generator(to_onto_class2text_path, batch_size=self.batch_size)
            nbest_dict = self.batch_alignment(dp["Class-Text"], to_batch_generator, flag=flag)
            return nbest_dict
            
    def batch_alignment(self, from_class_text, to_batch_generator, flag="SRC"):
        from_labels, from_len = Ontology.parse_class_text(from_class_text)
        j = 0
        nbest_scores_list = []
        nbest_indicees_list = []
        for to_batch in to_batch_generator:
            batch_label_pairs = []
            batch_lens = []
            # prepare a batch of label pairs for a given from-onto class 
            for _, to_class_dp in to_batch.iterrows():
                to_labels, to_len = Ontology.parse_class_text(to_class_dp["Class-Text"])
                batch_label_pairs += [[from_label, to_label] for to_label in to_labels for from_label in from_labels]
                batch_lens.append(to_len * from_len)
            # compute the classification scores
            batch_scores = self.classifier(batch_label_pairs)
            pooled_batch_scores = self.batch_pooling(batch_scores, batch_lens)
            print(len(pooled_batch_scores))
            nbest_scores, nbest_indices = torch.topk(pooled_batch_scores, k=self.nbest)
            nbest_indices += j * len(to_batch)
            nbest_scores_list.append(nbest_scores)
            nbest_indicees_list.append(nbest_indices)
        batch_nbest_scores, batch_nbest_indices = torch.topk(torch.stack(nbest_scores_list), k=self.nbest)
        return batch_nbest_scores, batch_nbest_indices
            
            
    def batch_pooling(self, batch_scores, batch_lens):
        assert self.strategy == "mean" or self.strategy == "max"
        pooling_fn = torch.max if self.strategy == "max" else torch.mean
        start = 0
        pooled_batch_scores = []
        for l in batch_lens:
            end = start + l
            pooled_batch_scores.append(pooling_fn(batch_scores[start: end]))
            start = end
        assert len(pooled_batch_scores) == len(batch_lens)
        return torch.stack(pooled_batch_scores)
            