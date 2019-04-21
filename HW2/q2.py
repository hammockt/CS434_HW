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
    """ normalizes all of the columns in the given matrix to [0,1] """
    return (matrix - matrix.min(0)) / matrix.ptp(0)

def total_wrong(expected_y, y):
    """ gives the percentage of wrong guesses in expected_y """
    diff = expected_y - y
    num_wrong = len([value for value in diff if not math.isclose(value, 0.0)])
    return num_wrong / len(y)

class Node():
    def __init__(self, x_vals, y_vals):
        self.children = []
        self.x_vals = x_vals
        self.y_vals = y_vals

    def entropy():
        y_counts = {}
        for y_val in self.y_vals:
            if y_val in y_counts:
                y_counts[y_val] = 0
            else:
                y_counts[y_val] += 1
        total_y_count = sum(y_counts.values())
        u_s = 0
        for count in y_counts.values():
            p_val = count / total_y_count
            u_s -= p_val * math.log2(p_val)

        for child in self.children:
            u_s -= len(child.y_vals) / total_y_count * child.entropy()
        return u_s

def main():
    """ entry point """
    if len(argv) != 3:
        print("Wrong number of arguments!")
        errmsg = f"Usage: python3 {argv[0]} <training file> <testing file>"
        sys.exit(errmsg)

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

    ###########################################################
    # classify/guess the y-values using our knn closest pairs #
    ###########################################################

    expected_training_y = [knn(training_x, training_y, point, k) for point in training_x]
    expected_validation_y = [knn(training_x, training_y, point, k, True) for point in training_x]
    expected_test_y = [knn(training_x, training_y, point, k) for point in test_x]

    ###################################
    # get the training and test error #
    ###################################

    print("training error:      {0:.2f}".format(total_wrong(expected_training_y, training_y)))
    print("leave-one-out error: {0:.2f}".format(total_wrong(expected_validation_y, training_y)))
    print("testing error:       {0:.2f}".format(total_wrong(expected_test_y, test_y)))

main()
