"""
Utility to help test random forests
"""

import sys
import glob
import parse_data
from decision_tree import DecisionTree
from random_forest import RandomForest

def main(argv):
	""" entry point to the program """

	if len(argv) != 3:
		sys.exit(f"Usage python3 {argv[0]} <testing_file> <tree_json_dir>")

	_, test_y, test_x = parse_data.read_data(argv[1], skip_header=False, delimiter=",")

	json_files = glob.glob(f"{argv[2]}/*.json")
	forest = RandomForest()
	for filename in json_files:
		with open(filename, "r") as tree_file:
			tree = DecisionTree.from_json(tree_file.read())
			forest.add_tree(tree)

	total_right = 0
	for i, point in enumerate(test_x):
		expected = forest.predict(point)
		if test_y[i] == expected:
			total_right += 1

	accuracy = total_right / len(test_y)
	print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
	main(sys.argv)
