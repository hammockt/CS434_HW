import sys
import glob
import os
import json
import numpy
import parse_data
from decision_tree import DecisionTree

numpy.set_printoptions(suppress=True)
numpy.set_printoptions(threshold=sys.maxsize)


def sigmoid(x):
    #return x ** 100
    return 1 / (1 + numpy.exp(-200*x + 136.25))

def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def dist(a, b):
    total = 0
    for i, item in enumerate(a):
        if item == b[i]:
            total += 1
    accuracy = total / len(a)
    return accuracy
    return numpy.linalg.norm(numpy.subtract(a, b))

def main(argv):
    if len(argv) != 3:
        sys.exit(f"Usage python3 {argv[0]} <testing_file> <tree_json_dir>")

    _, test_y, test_x = parse_data.read_data(argv[1], skip_header=False, delimiter=",")

    tree_files = glob.glob(f"{argv[2]}/*.json")
    forest = []
    weights_filename = f"{argv[2]}/tree_weights.json"
    if tree_files.count(weights_filename) > 0:
        tree_files.remove(weights_filename)
    for filename in tree_files:
        with open(filename, "r") as tree_file:
            tree = DecisionTree.from_json(tree_file.read())
            forest.append(tree)

    diffs = []
    forest_predictions = []
    weights = {}
    for i, tree in enumerate(forest):
        tree_predictions = [tree.predict(x) for x in test_x]
        forest_predictions.append(tree_predictions)

        diff = dist(test_y, tree_predictions)
        weights[tree_files[i]] = diff
        #diffs.append(diff)

    min_weight = min(weights.values())
    max_weight = max(weights.values())
    for item, val in weights.items():
        weights[item] = sigmoid(normalize(val, min_weight, max_weight))
    min_weight = min(weights.values())
    max_weight = max(weights.values())
    for item, val in weights.items():
        weights[item] = normalize(val, min_weight, max_weight)
    with open(weights_filename, "w") as weights_file:
        weights_file.write(json.dumps(weights))
    return
    sorted_refs = list(range(len(forest)))
    sorted_refs.sort(key=lambda ref: diffs[ref])
    #do not need diffs anymore
    del diffs

    prediction_sum = numpy.array(forest_predictions[sorted_refs[0]])
    smallest_dist = dist(test_y, prediction_sum)
    print(smallest_dist)
    best_trees = [tree_files[sorted_refs[0]]]
    for ref in sorted_refs[1:]:
        #correctness = numpy.subtract(forest_predictions[ref], test_y)
        #total_wrong = numpy.count_nonzero(correctness)
        #accuracy = 1 - (total_wrong / len(test_y))
        #print(f"Accuracy: {accuracy}")

        new_combination = numpy.add(prediction_sum, forest_predictions[ref])
        normalized_combination = new_combination / (len(best_trees) + 1)

        new_dist = dist(test_y, normalized_combination)
        #might need to make this an <= due to math
        if new_dist < smallest_dist:
            prediction_sum = new_combination
            smallest_dist = new_dist
            print(smallest_dist)
            best_trees.append(tree_files[ref])

    #should have the combination of trees that give us the closest value to the ground truth (i.e. the test data)
    #choose a set of trees such that the dist(ground, predict_prob) is minimized to 1
    #only problem is with multiple labels, this could be biased against labels that have larger distances (i.e. 2 and 0 vs 1 and 0)
    #print(prediction_sum / len(best_trees))
    for tree_file in tree_files:
        if tree_file not in best_trees:
            print(f"removing {tree_file}")
            os.remove(tree_file)

    #example:
    # ground [0, 0, 0, 0, 0, 1,   1,   1, 1,   1]
    # tree8  [0, 0, 0, 0, 0, 1,   1,   1, 0,   0]
    # tree7  [0, 0, 0, 0, 0, 1,   0,   0, 0,   1]
    # combin [0, 0, 0, 0, 0, 1, 0.5, 0.5, 0, 0.5]
    # dist(ground, tree8)  = 1.4142135623730951
    # dist(ground, combin) = 1.3228756555322954
    # therefore having tree8 and tree7 increases our accuracy/confidence

if __name__ == "__main__":
    main(sys.argv)
