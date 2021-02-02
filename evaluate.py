import argparse
from onto_align.eval import Evaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ontology Mappings.")
    parser.add_argument('-p', '--pre', type=str)
    parser.add_argument('-r', '--ref', type=str)
    args = parser.parse_args()
    evaluator = Evaluator(args.pre, args.ref)
    print("------ Results ------")
    print("P =", evaluator.P)
    print("R =", evaluator.R)
    print("F1 =", evaluator.F1)
