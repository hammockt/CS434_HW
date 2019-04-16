"""
implementation for part1
"""

import sys
import numpy

numpy.set_printoptions(suppress=True)

#normalize(x_i) = (x_i - min(x)) / range(x)
#gets the min and range for each column and uses that to normalize the column
def normalize_columns(matrix):
	""" normalizes all of the columns in the given matrix to [0,1] """
	return (matrix - matrix.min(0)) / matrix.ptp(0)

def calc_point_distance(matrix):
	""" create the data structures to determine the k closest points, from a point """
	#a hashtable of distances between two points, so we won't have to re-compute
	#key = tuple(point_1, point2), value = distance between the points
	distances = {}

	closest_pairs = []
	for i, point1 in enumerate(matrix):
		point_distances = []

		for j, point2 in enumerate(matrix):
			if i == j:
				continue

			key = (i, j)
			if key not in distances:
				distances[key] = distances[(j, i)] = numpy.linalg.norm(point1 - point2)

			distance = distances[key]
			point_distances.append((j, distance))

		point_distances.sort(key=lambda item: item[1])
		closest_pairs.append(point_distances)

	return closest_pairs

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

	#######################################################
	# create data structures for finding k closest points #
	#######################################################

	#point = row in x and y datasets
	#create a array
	#array = [point_1, point_2, ..., point_n]
	#	where array contains all other points that are sorted by distance from point
	#	this means that point_1 should be the closest to the point
	#advantages:
	#	can get k closest pairs in constant time
	#	this means we can get k closest pairs, n times in O(n)
	#	which may be nice when we need to do this for 51 different k
	#disadvantages:
	#	bare minimum O(n^2) to compute this, but I do have a O(n log n) from a 325 assignment
	#	space complexity of O(n^2)
	#	but we can save space by just storing the indexes of the points/rows
	#	we should really do this because it also makes looking up the corresponding y value trivial
	#		which we will need to done in the knn classification
	#possible bugs:
	#	don't put the point in it's own array
	#	if that happens then all arrays will contain their index as their first-ish element
	#	we can validate this by checking that the array does not contain it's own index
	training_closest_pairs = calc_point_distance(training_x)
	test_closest_pairs = calc_point_distance(test_x)

	###########################################################
	# classify/guess the y-values using our knn closest pairs #
	###########################################################

	#machine learning stuff
	#also the popular vote is flawed
	#so we should implement a electoral college system instead for our knn selection

	###################################
	# get the training and test error #
	###################################

	#diff = expected_y - y
	#total_wrong = where diff_i != 0
	#accuracy = total_wrong / total
	#WARNING: y is either -1 or 1

main()
