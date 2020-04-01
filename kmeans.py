import numpy as np
from numpy.linalg import norm


class Kmeans:
	"""
	Implementing Kmeans algorithm
	"""
	def __init__(self, n_clusters, max_iter=100, random_state=123):
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.random_state = random_state

	def initialize_centroids(self, X):
		centroids = X.copy()
		np.random.shuffle(centroids)
		return centroids[:self.n_clusters]

	def compute_centroids(self, X, labels):
		centroids = np.zeros((self.n_clusters, X.shape[1]))
		for k in range(self.n_clusters):
			centroids[k, :] = np.mean(X[labels == k, :], axis=0)
		return centroids

	def compute_distance(self, X, centroids):
		distance = np.zeros((X.shape[0], self.n_clusters))
		for k in range(self.n_clusters):
			row_norm = norm(X - centroids[k, :], axis=1)
			distance[:, k] = np.square(row_norm)
		return distance

	def find_closest_cluster(self, distance):
		return np.argmin(distance, axis=1)

	def compute_sse(self, X, labels, centroids):
		distance = np.zeros(X.shape[0])
		for k in range(self.n_clusters):
			distance[labels==k] = norm(X[labels == k] - centroids[k], axis=1)
		return np.sum(np.square(distance))

	def fit(self, X):
		# 1. initialize centroids
		self.centroids = self.initialize_centroids(X)
		# 2. iterate through, to update centroids until centroids no longer move
		for i in range(self.max_iter):
			old_centroids = self.centroids
			# calculate distance of each point to each centroid
			distance = self.compute_distance(X, old_centroids)
			# find the closest centroid
			self.labels = self.find_closest_cluster(distance)
			self.centroids = self.compute_centroids(X, self.labels)
			if np.all(old_centroids == self.centroids):
				break
		self.error = self.compute_sse(X, self.labels, self.centroids)

	def predict(self, X):
		distance = self.compute_distance(X, self.centroids)
		return self.find_closest_cluster(distance)



