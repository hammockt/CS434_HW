"""
implementation for part1
"""

import sys
import math
import heapq
import numpy

numpy.set_printoptions(suppress=True)

"""
normalize(x_i) = (x_i - min(x)) / range(x)
gets the min and range for each column and uses that to normalize the column
"""
def normalize_columns(matrix):
	""" normalizes all of the columns in the given matrix to [0,1] """
	return (matrix - matrix.min(0)) / matrix.ptp(0)

def distance_squared(point1, point2):
	""" calculates the distance^2 between two points """
	diff = point1 - point2
	return numpy.dot(diff, diff)

"""
leave-one-out cross validation is basically the point ignoring itself from our
	model when the matrix is the training data
rather than removing and readding the point, we can just ignore it
however since there can be duplicates of the point, we only want to ignore it once
"""
def knn(matrix_x, matrix_y, point, k, ignore_itself=False):
	""" predicts point's y value based off of it's closest members in matrix """
	point_distances = []
	for i, m_point in enumerate(matrix_x):
		if(ignore_itself and (m_point == point).all()):
			ignore_itself = False
			continue

		"""
		item[0] = negative distance
		item[1] = the y value of m_point
		also heapq can either sort comparables or the first value in a tuple
		i.e. no dictionaries unfortunately
		"""
		item = (-1*distance_squared(point, m_point), matrix_y[i])
		if len(point_distances) < k:
			heapq.heappush(point_distances, item)
		elif item[0] > point_distances[0][0]:
			heapq.heappushpop(point_distances, item)

	#mode of the k closest points
	y_values = {}
	for item in point_distances:
		if item[1] not in y_values:
			y_values[item[1]] = {"count": 0, "total_dist": 0, "value": item[1]}
		y_values[item[1]]["count"] += 1
		y_values[item[1]]["total_dist"] += item[0]

	"""
	in python3, dict.values() returns a view instead of a list
	i.e. we must use sorted() instead of dict.values().sort()
	"""
	y_values_list = sorted(y_values.values(),
	                       key=lambda y_value: (y_value["count"], y_value["total_dist"]),
	                       reverse=True)

	return y_values_list[0]["value"]

def total_wrong(expected_y, y):
	""" gives the percentage of wrong guesses in expected_y """
	diff = expected_y - y
	num_wrong = len([value for value in diff if not math.isclose(value, 0.0)])
	return num_wrong / len(y)

def main():
	""" entry point """
	if len(sys.argv) != 4:
		print("Usage: q1.py <training file> <testing file> k")
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
