# dict for large_bio ontology uris to prefixes
onto_uri_dict = {
    "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0": "fma",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl": "nci",
    "http://www.ihtsdo.org/snomed": "snomed"
}

onto_uri_dict_inv = {
    v: k for k, v in onto_uri_dict.items()
}
