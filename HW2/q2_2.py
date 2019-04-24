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

def build_tree(node, depth):
    if depth <= 0:
        return
    node.test_and_apply()
    for child in node.children:
        build_tree(child, depth - 1)

def main(argv):
    """ entry point """
    if len(argv) != 4:
        print("Wrong number of arguments!")
        sys.exit(f"Usage: python3 {sys.argv[0]} <training file> <testing file> <d>")

    ################
    # get the data #
    ################

    training_data = numpy.genfromtxt(argv[1], delimiter=",")
    #need all but the first column, could be better but I like the verbosity
    training_x = numpy.delete(numpy.copy(training_data), 0, axis=1)
    #need only the first column
    training_y = training_data[:, 0]

    test_data = numpy.genfromtxt(argv[2], delimiter=",")
    test_x = numpy.delete(numpy.copy(test_data), 0, axis=1)
    test_y = test_data[:, 0]

    d = int(argv[3])

    ##########################
    # normalize the features #
    ##########################

    training_x = normalize_columns(training_x)
    test_x = normalize_columns(test_x)

    basic_node = Node(training_x, training_y)

    build_tree(basic_node, d)

    expected_training_y = [basic_node.predicted_value(point) for point in training_x]
    expected_test_y = [basic_node.predicted_value(point) for point in test_x]
    print(f"d:              {d}")
    print("Training error: {0:.3f}".format(total_wrong(expected_training_y, training_y)))
    print("Testing error:  {0:.3f}".format(total_wrong(expected_test_y, test_y)))

main(sys.argv)
