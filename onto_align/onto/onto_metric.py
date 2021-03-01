from scipy.spatial.distance import cosine
from textdistance import levenshtein


class OntoMetric:
    
    def __init__(self):
        self.cos_dist = cosine  # 1 - cos(v_1, v_2)
        self.norm_edit_dist = levenshtein.normalized_distance  # edit-distance / max(v_1, v_2)
        

onto_metric = OntoMetric()