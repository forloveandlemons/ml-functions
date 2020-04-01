# input X, y

import numpy as np

def get_gradient(w, X, y):
    y_estimate = X.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    gradient = -(1.0/len(X)) * error.dot(X)
    return gradient, np.power(error, 2)

w = np.random.rand(3)
learning_rate = 0.5
tolerance = 1e-5

max_iter = 100
iteration = 0

train_X = np.arange(300).reshape(100, 3)
train_y = np.arange(100) - 5

while iteration < max_iter:
	gradient, error = get_gradient(w, train_X, train_y)
	new_w = w - learning_rate * gradient
	if np.sum(abs(new_w - w)) < tolerance:
		print("Converged.")
		break
	if iteration % 100 == 0:
		print("Iteration {} - Error {}".format(str(iteration), str(error)))
	iteration += 1
	w = new_w


