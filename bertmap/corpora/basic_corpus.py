import pandas as pd

class OntologyCorpus:
    
    def __init__(self, corpus_path=None):
        if not corpus_path:
            self.create_corpora()
        else:
            # load corpus from local storage
            self.load_corpora(save_dir=corpus_path)
        
    def create_corpora(self):
        raise NotImplementedError
        
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