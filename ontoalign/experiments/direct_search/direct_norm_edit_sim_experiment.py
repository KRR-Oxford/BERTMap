"""Direct Search Experiment on using the normalized edit distance score as the distance metric. 
   Unlike the BERT experiment where the batched (vectorization) algorithm is used, here we apply the multiprocessing on each batch.
"""

from ontoalign.onto import Ontology
from ontoalign.experiments import OntoAlignExperiment
from itertools import product
from textdistance import levenshtein
import multiprocessing
import time

class DirectNormEditSimExperiment(OntoAlignExperiment):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, save_path, 
                 src_batch_size=1000, tgt_batch_size=1000,
                 task_suffix="small"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, 
                         save_path, task_suffix=task_suffix, name="nes")
        
        self.src_batch_size = src_batch_size
        self.tgt_batch_size = tgt_batch_size
        self.src2tgt_mappings = None
        self.tgt2src_mappings = None
        self.combined_mappings = None
        
    # def run(self):
    #             t_start = time.time()
    #     self.alignment()
    #     t_end = time.time()
    #     t = t_end - t_start
    #     self.log_print('the program time is :%s' %t)
        

    def alignment(self, from_batch, _, flag="SRC"):
        to_onto_lexicon = self.tgt_onto_lexicon if flag == "SRC" else self.src_onto_lexicon
        batch_norm_edit_sim = lambda x, y: self.max_norm_edit_sim(Ontology.parse_entity_lexicon(x)[0], Ontology.parse_entity_lexicon(y)[0])
        to_entity_idx_list = []
        mapping_values = []
        for _, from_entity in from_batch.iterrows():
            from_lexicon = from_entity["Entity-Lexicon"]
            result = to_onto_lexicon["Entity-Lexicon"].apply(lambda to_lexicon: batch_norm_edit_sim(from_lexicon, to_lexicon), )
            to_entity_idx_list.append(result.idxmax())
            mapping_values.append(result.max())
        to_entity_iris = list(to_onto_lexicon.iloc[to_entity_idx_list]["Entity-IRI"])
        return to_entity_iris, mapping_values

    @staticmethod    
    def max_norm_edit_sim(from_lexicon, to_lexicon):
        label_pairs = product(from_lexicon, to_lexicon)
        sim_scores = [levenshtein.normalized_similarity(src, tgt) for src, tgt in label_pairs]
        return max(sim_scores)
    
    def save(self):
        self.src2tgt_mappings.to_csv(f"{self.save_path}/{self.src}2{self.tgt}.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')
        self.tgt2src_mappings.to_csv(f"{self.save_path}/{self.tgt}2{self.src}.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')
        self.combined_mappings.to_csv(f"{self.save_path}/{self.src}-{self.tgt}-combined.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')