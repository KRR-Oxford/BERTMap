from bertmap.onto import Ontology
import pandas as pd

class OntologyCorpus:
    
    def __init__(self, corpus_path=None):
        self.corpus_names = []
        self.onto_name = ""
        self.corpus_path = corpus_path
        
    def train_val_split(self, corpus_names):
        raise NotImplementedError

    def save_corpora(self, save_dir):
        for name in self.corpus_names:
            corpus = getattr(self, name)
            df = pd.DataFrame(corpus, columns=["Label1", "Label2", "Synonymous"])
            df.to_csv(f"{save_dir}/{self.onto_name}.{name}.tsv", sep='\t', index=False)
            
    def load_corpora(self, save_dir):
        print("--------------- Loaded Corpora Sizes --------------")
        for name in self.corpus_names:
            setattr(self, name, pd.read_csv(f"{save_dir}/{self.onto_name}.{name}.tsv", sep='\t'))
            tag = " ".join(name.split("_"))
            print(f"{tag}: {len(getattr(self, name))}")
        print("---------------------------------------------------")