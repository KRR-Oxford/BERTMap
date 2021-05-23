from owlready2 import get_ontology
import os
main_dir = os.getcwd().split("BERTMap")[0] + "BERTMap"
import sys
sys.path.append(main_dir)
import re
from bertmap.utils import uniqify, banner
from copy import deepcopy

choice = sys.argv[1]  # fma, nci, whole
banner(f"load snomed from Largebio (w.r.t. {choice})")
if choice == "fma" or choice == "nci":
    snomed = get_ontology(main_dir + f"/data/largebio/ontos/snomed2{choice}.small.owl").load()
elif choice == "whole":
    snomed = get_ontology(main_dir + "/data/largebio/ontos/snomed.whole.owl").load()
else:
    raise TypeError("choices are fma, nci and whole ...")
banner(f"load newest raw SNOMED")
snomed_ori = get_ontology(main_dir + "/data/largebio/ontos/SNOMED.ori.owl").load()

# remove (branch) for every label in Original SNOMED
banner("transform label (branch) -> label")
pa = r"^(.+) \(.+?\)$"  # label (branch) -> label
for cl in snomed_ori.classes():
    labs = []
    print(f"Before: [{cl}]: {cl.label}")
    for x in cl.label:
        labs.append(re.findall(pa, x)[0])
    if labs:
        cl.label = labs
    else:
        print(cl, cl.label)
        break
    print(f"After: [{cl}]: {cl.label}")

banner(f"construct snomed+ (w.r.t {choice})")
# add labels to snomed to form snomed+
empty = []
unique = 0
multiple = 0
added = 0
for cl in snomed.classes():
    labs = deepcopy(list(cl.label))
    print(f"Before: [{cl}]: {labs}")
    classes = []
    for x in cl.label:
        pref_cls = list(snomed_ori.search(prefLabel=x, _case_sensitive=False))
        alt_cls = list(snomed_ori.search(altLabel=x, _case_sensitive=False))
        lab_cls = list(snomed_ori.search(label=x, _case_sensitive=False))
        classes += pref_cls + alt_cls + lab_cls
        assert len(classes) >= len(pref_cls) + len(alt_cls) + len(lab_cls)
    classes = uniqify(classes)
    # remove classes from snomed instead of original SNOMED
    classes = [c for c in classes if "http://www.ihtsdo.org/snomed#" not in c.iri]
    print(f"Found classes: {classes}")

    if len(classes) == 0: empty.append(cl)
    elif len(classes) == 1: unique += 1
    elif len(classes) > 1: multiple += 1

    for c in classes:
        labs += list(c.label) + list(c.prefLabel) + list(c.altLabel)

    labs = uniqify(labs)
    if len(labs) > 1: 
        added += 1
        cl.label = labs
    print(f"After: [{cl}]: {cl.label}")

if choice == "fma" or choice == "nci":
    snomed.save(file=f"snomed+2{choice}.small.owl")
else:
    snomed.save(file=f"snomed+.whole.owl")