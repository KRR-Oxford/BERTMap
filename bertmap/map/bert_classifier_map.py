from bertmap.map import OntoMapping
from bertmap.bert import PretrainedBERT
from bertmap.onto import Ontology
from bertmap.utils import get_device
import torch
import time
import pandas as pd

class BERTClassifierMapping(OntoMapping):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_class2text_tsv, tgt_onto_class2text_tsv, 
                 save_path, batch_size, nbest=2, task_suffix="small", name="bc-tuned-mean", 
                 bert_path="", tokenizer_path="emilyalsentzer/Bio_ClinicalBERT", string_match=True, device_num=0):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_class2text_tsv, 
                         tgt_onto_class2text_tsv, save_path, task_suffix=task_suffix, name=name)    
        self.strategy = name.split("-")[2]  # ["bc", "tuned", "mean"]
        self.batch_size = batch_size   
        self.nbest = nbest
        self.string_match = string_match
        
        self.bert = PretrainedBERT(pretrained_path=bert_path, tokenizer_path=tokenizer_path, with_classifier=True)
        self.device = get_device(device_num=device_num)
        self.bert.model.to(self.device)
        
        self.tokenize = lambda x: self.bert.tokenizer(x, padding=True, return_tensors="pt")
        self.softmax = torch.nn.Softmax(dim=1)
        self.classifier = lambda x: self.softmax(self.bert.model(**x).logits)[:, 1]
        
    def align_config(self, flag="SRC"):
        assert flag == "SRC" or flag == "TGT"
        from_onto_class2text_path = self.src_onto_class2text_path
        to_onto_class2text_path = self.tgt_onto_class2text_path
        from_index = self.src_index
        to_index = self.tgt_index
        map_name = "src2tgt_mappings"
        if flag == "TGT":
            from_onto_class2text_path, to_onto_class2text_path = to_onto_class2text_path, from_onto_class2text_path
            from_index, to_index = to_index, from_index
            map_name = "tgt2src_mappings"
        return from_onto_class2text_path, to_onto_class2text_path, from_index, to_index, map_name
    
    def fixed_one_side_alignment(self, flag="SRC", start=0):
        # configurations
        self.start_time = time.time()
        from_onto_class2text_path, to_onto_class2text_path, _, to_index, map_name = self.align_config(flag=flag)
        from_onto_class2text = Ontology.load_class2text(from_onto_class2text_path)
        to_onto_class2text = Ontology.load_class2text(to_onto_class2text_path)
        # fix each from-onto class and calculate the mappings
        results = []
        for i, dp in from_onto_class2text.iterrows():
            # set up a start point to prevent unexpected intervention of the program
            if i < start:
                continue
            from_labels, from_len = Ontology.parse_class_text(dp["Class-Text"])
            # reduce search space if the sub-word level inverted index is provided
            search_space = to_onto_class2text if not to_index else self.select_candidates(dp["Class-Text"], flag=flag)
            if len(search_space) == 0:
                self.log_print("[Time: {round(time.time() - self.start_time)}][{self.name}][{print_flag}][#Class: {i}] No candidates available for for current entity ...")
                continue
            # normalize the batch size to prevent memory overflow while preserving the KTop functionality
            to_batch_size = max(self.batch_size // from_len, self.nbest + 1)  
            to_batch_generator = Ontology.class2text_batch_generator(search_space, batch_size=to_batch_size)
            nbest_results = self.batch_alignment(from_labels, from_len, to_batch_generator, to_batch_size, flag=flag)
            # collect the results
            for to_class_ind, mapping_score in nbest_results:
                if mapping_score <= 0.01:
                    mapping_score = 0.0
                to_class_iri = search_space.iloc[to_class_ind]["Class-IRI"]
                result = (dp["Class-IRI"], to_class_iri, mapping_score) if flag == "SRC" else (to_class_iri, dp["Class-IRI"], mapping_score)
                results.append(result)
                print_flag = f"{flag}: {self.src}" if flag == "SRC" else f"{flag}: {self.tgt}"
                self.log_print(f"[Time: {round(time.time() - self.start_time)}][{self.name}][{print_flag}][#Class: {i}][Mapping: {result}]")
        setattr(self, map_name, pd.DataFrame(results, columns=["Entity1", "Entity2", "Value"]))
            
    def batch_alignment(self, from_labels, from_len, to_batch_generator, to_batch_size, flag="SRC"):
        j = 0
        batch_nbest_scores = torch.tensor([-1] * self.nbest).to(self.device)
        batch_nbest_indices = torch.tensor([-1] * self.nbest).to(self.device)
        for to_batch in to_batch_generator:
            batch_label_pairs = []
            batch_lens = []
            # prepare a batch of label pairs for a given from-onto class 
            for m, to_class_dp in to_batch.iterrows():
                to_labels, to_len = Ontology.parse_class_text(to_class_dp["Class-Text"])
                label_pairs = [[from_label, to_label] for to_label in to_labels for from_label in from_labels]
                # return the map if the to-class has a label that is exactly the same as one of the labels of the from-class
                if self.string_match:
                    for pair in label_pairs:
                        if pair[0] == pair[1]:
                            return [(m, 1.0)]
                batch_label_pairs += label_pairs
                batch_lens.append(to_len * from_len)
            # compute the classification scores
            with torch.no_grad():
                model_inputs_dict = self.tokenize(batch_label_pairs)
                for k in model_inputs_dict.keys():
                    model_inputs_dict[k] = model_inputs_dict[k].to(self.device)
                batch_scores = self.classifier(model_inputs_dict)
                pooled_batch_scores = self.batch_pooling(batch_scores, batch_lens)
                K = len(pooled_batch_scores) if len(pooled_batch_scores) < self.nbest else self.nbest
                nbest_scores, nbest_indices = torch.topk(pooled_batch_scores, k=K)
                nbest_indices += j * to_batch_size
                # we do the substituion for every batch to prevent from memory overflow
                batch_nbest_scores, temp_indices = torch.topk(torch.cat([batch_nbest_scores, nbest_scores]), k=self.nbest)
                batch_nbest_indices = torch.cat([batch_nbest_indices, nbest_indices])[temp_indices]
                # print(f"[batch {j}] current time: {time.time() - self.start_time}")
                j += 1
        return list(zip(batch_nbest_indices.cpu().detach().numpy(), batch_nbest_scores.cpu().detach().numpy()))
            
    def batch_pooling(self, batch_scores, batch_lens):
        seq_of_scores = torch.split(batch_scores, split_size_or_sections=batch_lens)
        assert self.strategy == "mean" or self.strategy == "max"
        pooling_fn = torch.max if self.strategy == "max" else torch.mean
        pooled_batch_scores = [pooling_fn(chunk) for chunk in seq_of_scores]
        return torch.stack(pooled_batch_scores)
            