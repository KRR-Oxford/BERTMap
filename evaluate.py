import argparse
from onto_align.onto import OntoEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ontology Mappings.")
    parser.add_argument('-p', '--pre', type=str)
    parser.add_argument('-r', '--ref', type=str)
    parser.add_argument('-e', '--exc', type=str, default=None)
    args = parser.parse_args()
    evaluator = OntoEvaluator(args.pre, args.ref, args.exc)
    print("------ Results ------")
    print("P =", evaluator.P)
    print("R =", evaluator.R)
    print("F1 =", evaluator.F1)
