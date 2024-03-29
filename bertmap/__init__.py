import pandas as pd
from collections import defaultdict
from typing import Optional

# exclude mistaken parsing of string "null" to NaN
na_vals = pd.io.parsers.readers.STR_NA_VALUES.difference({"NULL", "null", "n/a"})

main_dir = __file__.replace("__init__.py", "")

namespaces = defaultdict(str)

# largebio
namespaces["http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#"] = "fma:"
namespaces["http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"] = "nci:"
namespaces["http://www.ihtsdo.org/snomed#"] = "snomed:"
# namespaces["http://snomed.info/id/"] = "snomed+:"  # most recent version of SNOMED (accessed on May 2021)

# phenotype
namespaces["http://purl.obolibrary.org/obo/"] = "obo:"
namespaces["http://www.orpha.net/ORDO/"] = "ordo:"



