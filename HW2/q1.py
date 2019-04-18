"""
implementation for part1
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

def distance_squared(point1, point2):
	""" calculates the distance^2 between two points """
	diff = point1 - point2
	return numpy.dot(diff, diff)

#leave-one-out cross validation is basically the point ignoring itself from our
#	model when the matrix is the training data
#rather than removing and readding the point, we can just ignore it
#however since there can be duplicates of the point, we only want to ignore it once
def knn(matrix_x, matrix_y, point, k, ignore_itself=False):
	""" predicts point's y value based off of it's closest members in matrix """
	point_distances = []
	for i, m_point in enumerate(matrix_x):
		if(ignore_itself and (m_point == point).all()):
			ignore_itself = False
			continue

		item = (-distance_squared(point, m_point), matrix_y[i])
		point_distances.append(item)

	#sort by the items distance
	point_distances.sort(key=lambda item: item[0], reverse=True)

	#mode of the k closest points
	y_vals = {item[1]: (0, 0) for item in point_distances[:k]}
	for item in point_distances[:k]:
		y_vals[item[1]] = (y_vals[item[1]][0] + 1, y_vals[item[1]][1] + item[0])
	y_vals_list = [(y_val, y_vals[y_val][0], y_vals[y_val][1]) for y_val in y_vals]
	y_vals_list.sort(key=lambda x: (x[1], x[2]), reverse=True)
	return y_vals_list[0][0]

def total_wrong(expected_y, y):
	""" gives the percentage of wrong guesses in expected_y """
	diff = expected_y - y
	num_wrong = len([value for value in diff if not math.isclose(value, 0.0)])
	return num_wrong / len(y)

def main():
	""" entry point """
	if len(sys.argv) != 4:
		print("Usage: q1.py train.txt k")
		sys.exit()

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

	#will come from sys.argv as a string...
	k = int(sys.argv[3])

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
