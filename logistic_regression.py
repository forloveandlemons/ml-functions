import numpy as np

max_iters = 10000

X = np.arange(300).reshape((100, 3))
label = np.arange(100)

weights = np.random.rand(X.shape[1])
tolerance = 1e-5


# logistic regression: same idea. but switch the loss function
def cost_function(X, labels, weights):
	N = len(labels)
	predictions = predict(X, weights)
	class1_cost = -label * np.log(predictions)
	class2_cost = -(1 - labels) * np.log(1 - predictions)
	cost = class1_cost + class2_cost
	return cost.sum() / N

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def predict(weights, X):
	z = np.dot(X, weights)
	return sigmoid(z)

cost_history = []
for i in range(max_iters):
	predictions = predict(weights, X)
	error = predictions - labels
	gradient = -(1.0/len(X)) * np.dot(X.T, error)
	new_weights = weights - learning_rate * weights
	cost = cost_function(X, labels, weights)
	cost_history.append(cost)
	if i % 100 == 0:
		print(i, cost)

