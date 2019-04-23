"""
implementation for part2
"""

import sys
import math
import statistics
import numpy

numpy.set_printoptions(suppress=True)

#normalize(x_i) = (x_i - min(x)) / range(x)
#gets the min and range for each column and uses that to normalize the column
def normalize_columns(matrix):
    """ Normalizes all of the columns in the given matrix to [0,1] """
    return (matrix - matrix.min(0)) / matrix.ptp(0)

def total_wrong(expected_y, y):
    """ Gives the percentage of wrong guesses in expected_y """
    diff = expected_y - y
    num_wrong = len([value for value in diff if not math.isclose(value, 0.0)])
    return num_wrong / len(y)


class Tree():
    def __init__(self, root):
        self.root = root
        self.height = self.get_height()

    def get_height(self, node_param=None):
        if node_param is None:
            node = self.root
        else:
            node = node_param
        if not node.children:
            return 0
        return 1 + max(self.get_height(node.children[0]),
                       self.get_height(node.children[1]))

    def predicted_value(self, point):
        return self.root.predicted_value(point)


class Node():
    """ represents a node in a decision tree """
    def __init__(self, x_vals, y_vals, parent=None):
        self.parent = parent
        self.children = []
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.feature = None
        self.test_bound = None

    #https://stackoverflow.com/a/1859910
    def entropy(self):
        """ calculates the entropy (uncertainty/randomness) of the node """
        y_counts = {}
        for y_val in self.y_vals:
            #python truncates floats if they are used as keys...
            #play it safe and use the exact string representation as the key instead (str also truncates)
            key = repr(y_val)
            if key not in y_counts:
                y_counts[key] = 0
            y_counts[key] += 1
        total_y_count = sum(y_counts.values())

        #we should probably break this into two separate methods...ehh
        u_s = 0
        if not self.children:
            for count in y_counts.values():
                p_val = count / total_y_count
                u_s -= p_val * math.log2(p_val)
        else:
            for child in self.children:
                u_s += len(child.y_vals) / total_y_count * child.entropy()

        return u_s

    def test(self):
        """ tests the node to figure out the best way to split the data """
        best_feature = {}
        for col in range(len(self.x_vals[0])):
            # Get column from x_vals
            feature_vals = [
                (y_index, self.x_vals[y_index][col])
                for y_index in range(len(self.y_vals))
            ]
            feature_vals.sort(key=lambda x: x[1])
            for index, val in enumerate(feature_vals[1:]):
                # Since we're skipping the first element
                j = index + 1
                # <, >=
                # Set y_val separations based on test boundary
                y_vals = {"l": [self.y_vals[k[0]] for k in feature_vals[:j]],
                          "r": [self.y_vals[k[0]] for k in feature_vals[j:]]}

                self.children.append(Node(None, y_vals["l"], self))
                self.children.append(Node(None, y_vals["r"], self))
                entropy = self.entropy()
                if not best_feature or best_feature["entropy"] > entropy:
                    best_feature = {
                        "feature_index": col,
                        "test_bound": val[1],
                        "entropy": entropy
                    }
                self.children = []
        self.feature = best_feature["feature_index"]
        self.test_bound = best_feature["test_bound"]
        return best_feature

    def test_and_apply(self):
        """ tests the node and applies the split """
        best_feature = self.test()
        x_vals = ([], [])
        y_vals = ([], [])
        for i, point in enumerate(self.x_vals):
            if point[self.feature] < self.test_bound:
                x_vals[0].append(point)
                y_vals[0].append(self.y_vals[i])
            else:
                x_vals[1].append(point)
                y_vals[1].append(self.y_vals[i])
        for i, x_val in enumerate(x_vals):
            self.children.append(Node(x_val, y_vals[i], self))
        return best_feature

    def predicted_value(self, point):
        """ descends the decision tree to predict the output of a point """
        if not self.children:
            return self.decision()

        if point[self.feature] < self.test_bound:
            return self.children[0].predicted_value(point)
        return self.children[1].predicted_value(point)

    def decision(self):
        """ determines how the output is picked on a leaf node """
        try:
            return statistics.mode(self.y_vals)
        except statistics.StatisticsError:
            return self.y_vals[0]


def main():
    """ entry point """
    if len(sys.argv) != 3:
        print("Wrong number of arguments!")
        sys.exit(f"Usage: python3 {sys.argv[0]} <training file> <testing file>")

    ################
    # get the data #
    ################

    training_data = numpy.genfromtxt(sys.argv[1], delimiter=",")
    #need all but the first column, could be better but I like the verbosity
    training_x = numpy.delete(numpy.copy(training_data), 0, axis=1)
    #need only the first column
    training_y = training_data[:, 0]

    test_data = numpy.genfromtxt(sys.argv[2], delimiter=",")
    test_x = numpy.delete(numpy.copy(test_data), 0, axis=1)
    test_y = test_data[:, 0]

    ##########################
    # normalize the features #
    ##########################

    training_x = normalize_columns(training_x)
    test_x = normalize_columns(test_x)

    basic_node = Node(training_x, training_y)
    entropy_before = basic_node.entropy()

    test_info = basic_node.test()
    feature_index = test_info["feature_index"]
    test_bound = test_info["test_bound"]
    entropy_after = test_info["entropy"]
    info_gain = entropy_before - entropy_after

    print(f"Feature index of test (decision stump): {feature_index}")
    print(f"Test boundary (decision stump):         {test_bound}")
    print(f"Information gain:                       {info_gain}")

    expected_training_y = [basic_node.predicted_value(point) for point in training_x]
    expected_test_y = [basic_node.predicted_value(point) for point in test_x]
    print("Training error:      {0:.2f}".format(total_wrong(expected_training_y, training_y)))
    print("Testing error:       {0:.2f}".format(total_wrong(expected_test_y, test_y)))

main()
