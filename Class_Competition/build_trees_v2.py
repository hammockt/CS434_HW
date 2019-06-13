"""
Utility to help build decision trees indefinitely
"""

import sys
import uuid
import random
import parse_data
from decision_tree import DecisionTree

def main(argc, argv):
    """ entry point to the program """

    if argc < 3 or argc > 4:
        sys.exit(f"Usage python3 {argv[0]} <training_file> <output_dir> <random_features?>")

    _, training_y, training_x = parse_data.read_data(argv[1], skip_header=False, delimiter=",")

    random_features = None
    if argc >= 4:
        random_features = int(argv[3])

    num_rows = len(training_y)
    while True:
        tree = DecisionTree()

        rows_to_evaluate = random.choices(range(num_rows), k=num_rows)
        tree.train(rows_to_evaluate, training_x, training_y, random_features=random_features)

        uuid_name = uuid.uuid4()
        filename = f"{argv[2]}/{uuid_name}.json"
        with open(filename, "w") as out_file:
            out_file.write(tree.to_json())
        print(filename)

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
