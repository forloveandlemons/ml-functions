import numpy as np
from numpy.linalg import norm

"""
K-Means steps:
1. initialize random centroids
2. for iteration in number of iterations do the following
	2.1 calculate each point's distance to each centroid
	2.2 find the closest centroid to each point (now each point has a new label)
	2.3 given the new labeling, calculate the new clustering centroids
	2.4 if the new centroids are the same as the old ones break
	2.5 calculate the sse (label & predicted label for each point)
"""

n_clusters = 10
max_iter = 10000

X = np.arange(300).reshape((100, 3))

# initialize centroids
def initialize_centroids(X):
	centroids = X.copy()
	np.random.shuffle(centroids)
	return centroids[:n_clusters]

centroids = initialize_centroids(X)

def compute_distance(X, centroids):
	distance = np.zeros((X.shape[0], n_clusters))
	for k in range(n_clusters):
		row_norm = norm(X - centroids[k, :], axis=1)
		distance[:, k] = np.square(row_norm)
	return distance

def find_closest_cluster(distance):
	return np.argmin(distance, axis=1)

def compute_centroids(X, labels):
	centroids = np.zeros((n_clusters, X.shape[1]))
	for k in range(n_clusters):
		centroids[k, :] = np.mean(X[labels == k, :], axis=0)
	return centroids

def compute_sse(X, labels, centroids):
	distance = np.zeros(X.shape[0])
	for k in range(n_clusters):
		distance[labels==k] = norm(X[labels == k] - centroids[k], axis=1)
	return np.sum(np.square(distance))

def fit(X, max_iter, centroids):
	centroids = initialize_centroids(X)
	for i in range(max_iter):
		old_centroids = centroids
		distance = compute_distance(X, old_centroids)
		labels = find_closest_cluster(distance)
		centroids = compute_centroids(X, labels)
		if np.all(centroids == old_centroids):
			break
	error = compute_sse(X, labels, centroids)
	return centroids

centroids = fit(X, max_iter, centroids)
def predict(X):
	distance = compute_distance(X, centroids)
	return find_closest_cluster(distance)

predict(X)

