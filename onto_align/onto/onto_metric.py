from scipy.spatial.distance import cosine
from textdistance import levenshtein
from itertools import product


class OntoMetric:
    
    cos_dist = cosine  # 1 - cos(v_1, v_2)
    norm_edit_dist = levenshtein.normalized_distance  # edit-distance / max(v_1, v_2)
    norm_edit_sim = levenshtein.normalized_similarity
        
    @classmethod    
    def min_norm_edit_dist(cls, src_lexicon, tgt_lexicon):
        label_pairs = product(src_lexicon, tgt_lexicon)
        dist_list = [cls.norm_edit_dist(src, tgt) for src, tgt in label_pairs]
        return min(dist_list)
    
    @classmethod    
    def max_norm_edit_sim(cls, src_lexicon, tgt_lexicon):
        label_pairs = product(src_lexicon, tgt_lexicon)
        sim_list = [cls.norm_edit_sim(src, tgt) for src, tgt in label_pairs]
        return max(sim_list)
