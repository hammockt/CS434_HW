"""
implementation for part2
"""

import sys
import math
import numpy
from node import Node

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

    test_info = basic_node.test_and_apply()
    feature_index = test_info["feature_index"]
    test_bound = test_info["test_bound"]
    entropy_after = test_info["entropy"]
    info_gain = entropy_before - entropy_after
    left_classify = int(basic_node.children[0].decision())
    right_classify = int(basic_node.children[1].decision())

    print(f"Feature index of test (decision stump): {feature_index}")
    print(f"Test boundary (decision stump):         {test_bound}")
    print(f"Classification of left child:           {left_classify}")
    print(f"Classification of right child:          {right_classify}")
    print(f"Information gain:                       {info_gain}")

    expected_training_y = [basic_node.predicted_value(point) for point in training_x]
    expected_test_y = [basic_node.predicted_value(point) for point in test_x]
    print("Training error:      {0:.3f}".format(total_wrong(expected_training_y, training_y)))
    print("Testing error:       {0:.3f}".format(total_wrong(expected_test_y, test_y)))

main()
