"""
Utility to help test random forests
"""

import sys
import glob
import parse_data
import json
from decision_tree import DecisionTree
from random_forest import RandomForest

def main(argv):
    """ entry point to the program """

    if len(argv) != 3:
        sys.exit(f"Usage python3 {argv[0]} <testing_file> <tree_json_dir>")

    _, test_y, test_x = parse_data.read_data(argv[1], skip_header=False, delimiter=",")

    json_files = glob.glob(f"{argv[2]}/*.json")
    forest = RandomForest()
    weights_filename = f"{argv[2]}/tree_weights.json"
    json_files.remove(weights_filename)
    weights = {}
    with open(weights_filename, "r") as weights_file:
        weights = json.loads(weights_file.read())
    for filename in json_files:
        with open(filename, "r") as tree_file:
            tree = DecisionTree.from_json(tree_file.read())
            forest.add_tree(tree)
        forest.weights.append(weights[filename])

    total_right = 0
    for i, point in enumerate(test_x):
        expected = forest.predict(point)
        if test_y[i] == expected:
            total_right += 1

    accuracy = total_right / len(test_y)
    print(f"Accuracy: {accuracy}")
    print(total_right, "out of", len(test_y))

if __name__ == "__main__":
    main(sys.argv)
