import sys
import glob
import parse_data
from decision_tree import DecisionTree
from random_forest import RandomForest

def main(argv):
	if len(argv) != 3:
		sys.exit(f"Usage python3 {argv[0]} <testing_file> <tree_json_dir>")

	rna_ids, _, test_x = parse_data.read_data(argv[1], skip_header=False, delimiter=",")

	json_files = glob.glob(f"{argv[2]}/*.json")
	forest = RandomForest()
	for filename in json_files:
		with open(filename, "r") as tree_file:
			tree = DecisionTree.from_json(tree_file.read())
			forest.add_tree(tree)

	for i, x in enumerate(test_x):
		prediction, confidence = forest.predict_with_confidence(x)

		if prediction == 0.0:
			confidence = 1 - confidence

		print(f"{rna_ids[i]},{confidence}")

if __name__ == "__main__":
	main(sys.argv)
