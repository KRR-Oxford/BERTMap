from bertmap.onto import Ontology
from bertmap.utils.oaei_utils import read_tsv_mappings

class ToBeAlignedOntologies:
    
    def __init__(self, src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, ref_mappings_tsv):
        self.src_onto_lexicon = Ontology.load_class2text(src_onto_lexicon_tsv)
        self.tgt_onto_lexicon = Ontology.load_class2text(tgt_onto_lexicon_tsv)
        self.ref_mappings = read_tsv_mappings(ref_mappings_tsv)
        
    def extract_alias_pairs_from_ref_mappings(self):
        for ref_map in self.ref_mappings:
            # src_class = 
            pass