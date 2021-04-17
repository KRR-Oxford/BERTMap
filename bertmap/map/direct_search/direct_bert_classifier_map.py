from bertmap.map.direct_search import DirectSearchMapping
from bertmap.bert import PretrainedBERT
from bertmap.onto import Ontology
from bertmap.utils import get_device
import torch
import time

class DirectBERTClassifierMapping(DirectSearchMapping):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_class2text_tsv, tgt_onto_class2text_tsv, save_path, 
                 batch_size, nbest=2, task_suffix="small", name="bc-tuned-mean", 
                 bert_path="", tokenizer_path="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_class2text_tsv, 
                         tgt_onto_class2text_tsv, save_path, task_suffix=task_suffix, name=name)    
        self.strategy = name.split("-")[2]  # ["bc", "tuned", "mean"]
        self.batch_size = batch_size   
        self.nbest = nbest
        self.bert = PretrainedBERT(pretrained_path=bert_path, tokenizer_path=tokenizer_path, with_classifier=True)
        self.device = get_device()
        self.bert.model.to(self.device)
        
        self.tokenize = lambda x: self.bert.tokenizer(x, padding=True, return_tensors="pt")
        self.softmax = torch.nn.Softmax(dim=1)
        self.classifier = lambda x: self.softmax(self.bert.model(**x).logits)[:, 1]
        
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
        self.start_time = time.time()
        from_onto_class2text_path, to_onto_class2text_path, mappings = self.align_config(flag=flag)
        from_onto_class2text = Ontology.load_class2text(from_onto_class2text_path)
        to_onto_class2text = Ontology.load_class2text(to_onto_class2text_path)
        for i, dp in from_onto_class2text.iterrows():
            to_batch_generator = Ontology.class2text_batch_generator(to_onto_class2text_path, batch_size=self.batch_size)
            nbest_results = self.batch_alignment(dp["Class-Text"], to_batch_generator, flag=flag)
            for to_class_ind, mapping_score in nbest_results:
                to_class_iri = to_onto_class2text.iloc[to_class_ind]["Class-IRI"]
                mappings.iloc[i] = [dp["Class-IRI"], to_class_iri, mapping_score] if flag == "SRC" \
                    else [to_class_iri, dp["Class-IRI"], mapping_score]
                self.log_print(f"[{self.name}][{flag}: {self.src}][#Entity: {i}][Mapping: {list(mappings.iloc[i])}]" if flag == "SRC" \
                    else f"[{self.name}][{flag}: {self.tgt}][#Entity: {i}][Mapping: {list(mappings.iloc[i])}]")
            
    def batch_alignment(self, from_class_text, to_batch_generator, flag="SRC"):
        from_labels, from_len = Ontology.parse_class_text(from_class_text)
        j = 0
        nbest_scores_list = []
        nbest_indices_list = []
        for to_batch in to_batch_generator:
            batch_label_pairs = []
            batch_lens = []
            # prepare a batch of label pairs for a given from-onto class 
            for _, to_class_dp in to_batch.iterrows():
                to_labels, to_len = Ontology.parse_class_text(to_class_dp["Class-Text"])
                batch_label_pairs += [[from_label, to_label] for to_label in to_labels for from_label in from_labels]
                batch_lens.append(to_len * from_len)
            # compute the classification scores
            with torch.no_grad():
                model_inputs_dict = self.tokenize(batch_label_pairs)
                for k in model_inputs_dict.keys():
                    model_inputs_dict[k] = model_inputs_dict[k].to(self.device)
                batch_scores = self.classifier(model_inputs_dict)
                pooled_batch_scores = self.batch_pooling(batch_scores, batch_lens)
                nbest_scores, nbest_indices = torch.topk(pooled_batch_scores, k=self.nbest)
                nbest_indices += j * self.batch_size
                nbest_scores_list.append(nbest_scores.cpu())
                nbest_indices_list.append(nbest_indices.cpu())
            # print(f"[batch {j}] current time: {time.time() - self.start_time}")
            j += 1
        final_nbest_scores, final_nbest_indices = torch.topk(torch.cat(nbest_scores_list), k=self.nbest)
        final_nbest_indices = torch.cat(nbest_indices_list)[final_nbest_indices]
        self.log_print(f"[Last batch] current time: {time.time() - self.start_time}")
        return list(zip(final_nbest_indices.detach().numpy(), final_nbest_scores.detach().numpy()))
            
            
    def batch_pooling(self, batch_scores, batch_lens):
        seq_of_scores = torch.split(batch_scores, split_size_or_sections=batch_lens)
        assert self.strategy == "mean" or self.strategy == "max"
        pooling_fn = torch.max if self.strategy == "max" else torch.mean
        pooled_batch_scores = [pooling_fn(chunk) for chunk in seq_of_scores]
        return torch.stack(pooled_batch_scores)
            