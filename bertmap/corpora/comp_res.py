"""
Ontology Corpus class from complementary resources.

The strategy is to add synonyms and nonsynonyms extracted from the complementary resource into the pre-built corpus in json format
"""

from bertmap.corpora import OntologyCorpus

class ComplementaryResourceCorpus(OntologyCorpus):
    
    def __init__(self, base_onto_corpus, complementary_resource, corpus_path):
        raise NotImplementedError