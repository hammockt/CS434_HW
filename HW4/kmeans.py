"""
Implementation for part 2 of the assignment
"""

import sys
import random
import numpy

def distance(point1, point2):
	""" calculates distance """
	diff = numpy.subtract(point1, point2)
	return numpy.dot(diff, diff)

def sse(array, point):
	""" calculates sum of squared errors """
	return numpy.sum(numpy.square(numpy.subtract(array, point)))

class Cluster():
	""" class to represent a given cluster """
	def __init__(self):
		# vector that represents the center of the cluster
		self.centroid = None

		# objective for learning, want to minimize this
		self.objective = float("inf")

		# points that fall into this cluster
		self.points = []

	def update_centroid(self):
		""" updates the centroid, the center point of a cluster """
		# mean of all of the points, else keep the old value
		if self.points:
			"""
			if you do not specify axis=0, numpy flattens the points
			which returns a number instead of a vector
			"""
			self.centroid = numpy.mean(self.points, axis=0)

	def update_objective(self):
		""" updates our objective that we want to minimze, which is sse """
		if self.points:
			if self.centroid is None:
				raise RuntimeError("Centroid is None")
			self.objective = sse(self.points, self.centroid)

class KMeansCluster():
	""" class that implements k means clustering """
	def __init__(self, k):
		self.k = k
		self.clusters = [Cluster() for i in range(k)]
		self.total_objective = float("inf")

	def nearest_cluster(self, point):
		""" returns the index of the cluster with the nearest centroid """
		nearest_centroid = 0
		nearest_distance = float("inf")

		# TODO: can technically start at the second cluster, we are closest to the first by default
		for i, cluster in enumerate(self.clusters):
			centroid_distance = distance(cluster.centroid, point)

			if centroid_distance < nearest_distance:
				nearest_centroid = i
				nearest_distance = centroid_distance

		return nearest_centroid

	def kmeans(self, data):
		""" clusters the data into k partitions """

		# randomly place the centroids
		random_indexes = random.sample(range(len(data)), self.k)
		for i, random_index in enumerate(random_indexes):
			self.clusters[i].centroid = data[random_index]

		#clustering algorithm
		num_iterations = 0
		while num_iterations < 50:
			# make sure the clusters are empty before we reassign
			for cluster in self.clusters:
				cluster.points = []

			# add the point to the nearest cluster
			for point in data:
				cluster_index = self.nearest_cluster(point)
				self.clusters[cluster_index].points.append(point)

			self.total_objective = 0
			for cluster in self.clusters:
				cluster.update_centroid()
				cluster.update_objective()
				self.total_objective += cluster.objective
			print(self.total_objective)

			num_iterations += 1

def main():
	""" entry point """
	if len(sys.argv) != 2:
		print("Usage: kmeans.py k")
		sys.exit()

	################
	# get the data #
	################

	training_data = numpy.genfromtxt("p4-data.txt", delimiter=",")
	k = int(sys.argv[1])

	k_clusters = KMeansCluster(k)
	k_clusters.kmeans(training_data)

main()
