"""
implementation for part2
"""

import sys
import math
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

    #https://stackoverflow.com/a/1859910
    def entropy(self):
        y_counts = {}
        for y_val in self.y_vals:
            #python truncates floats if they are used as keys...
            #play it safe and use the exact string representation as the key instead (str also truncates)
            key = repr(y_val)
            if key not in y_counts:
                y_counts[key] = 0
            else:
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

def main():
    """ entry point """
    if len(sys.argv) != 3:
        print("Wrong number of arguments!")
        errmsg = f"Usage: python3 {sys.argv[0]} <training file> <testing file>"
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

    basic_node = Node(training_x, training_y)
    entropy_before = basic_node.entropy()
    print(f"entropy of root: {entropy_before}")

    #basic 50/50 split
    basic_node.children.append(Node(training_x[:len(training_x)//2], training_y[:len(training_y)//2]))
    basic_node.children.append(Node(training_x[len(training_x)//2:], training_y[len(training_y)//2:]))
    entropy_after = basic_node.entropy()
    print(f"entropy of basic split: {entropy_after}")
    print(f"information gain: {entropy_before - entropy_after}")
    print(f"entropy of child 0: {basic_node.children[0].entropy()}")
    print(f"entropy of child 1: {basic_node.children[1].entropy()}")

main()
