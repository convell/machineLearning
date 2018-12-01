import numpy as np
from sklearn.datasets import make_moons

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def calculate_loss(X, y, model={"W1":np.array([1,2]),"W2":np.array([2,3]),"b1":4,"b2":4}):
	loss = 0
	sample_counter = 0

	for sample in X:
		a_activation = (sample * model["W1"]) + model["b1"]
		h_activation = np.tanh(a_activation)
		z_activation = (h_activation * model["W2"]) + model["b2"]
		yHat = softmax(z_activation)
		loss += np.log(yHat[y[sample_counter]])
		sample_counter += 1

	loss = -(1/sample_counter) * loss

	return loss

def predict(x, model={"W1":np.array([1,2]),"W2":np.array([2,3]),"b1":4,"b2":4}):
	a_activation = (x * model["W1"]) + model["b1"]
	h_activation = np.tanh(a_activation)
	z_activation = (h_activation * model["W2"]) + model["b2"]
	yHat = softmax(z_activation)
	return np.argmax(yHat)

X, y = make_moons(200 , noise =0.20)
print(calculate_loss(X,y))
print(predict(X[4]))
print(y[4])
