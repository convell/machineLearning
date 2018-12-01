import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def calculate_loss(X, y, model={"W1":np.array([1,2]),"W2":np.array([2,3]),"b1":4,"b2":4}):
	loss = 0
	sample_counter = 0
	print(model)
	for sample in X:
		a_activation = (sample * model["W1"]) + model["b1"]
		h_activation = np.tanh(a_activation)
		z_activation = (h_activation * model["W2"]) + model["b2"]
		yHat = softmax(z_activation)
		loss += np.log(yHat[y[sample_counter]])
		sample_counter += 1

	loss = -(1/sample_counter) * loss

	return loss

def predict(x, model):
	a_activation = (x * model["W1"]) + model["b1"]
	h_activation = np.tanh(a_activation)
	z_activation = (h_activation * model["W2"]) + model["b2"]
	yHat = softmax(z_activation)
	return np.argmax(yHat), a_activation, h_activation, z_activation, yHat

def good_predict(x, model):
	a_activation = (x * model["W1"]) + model["b1"]
	h_activation = np.tanh(a_activation)
	z_activation = (h_activation * model["W2"]) + model["b2"]
	yHat = softmax(z_activation)
	return np.argmax(yHat, axis=1)

def build_model(X,y,nn_hdim, num_passes=200000,print_loss=False):
	#intialize weights
	model = {"W1":np.array([0,0]),"W2":np.array([0,0]),"b1":0,"b2":0}

	for weight_name, weight_value in model.items():

		if type(weight_value) == type(np.array([0,0])): #if weight array
			counter = 0
			for x in weight_value:
				model[weight_name][counter] = np.random.normal(loc=0.0,scale=1.0)
				counter += 1

		else: #if bias
			model[weight_name] = np.random.normal(loc=0.0,scale=1.0)
			print(weight_value)

	for epoch in range(num_passes):
		sample_counter = 0
		for sample in X:
			prediction, a_activation, h_activation, z_activation, yHat = predict(sample,model)
			model["W1"] = x * (1 - np.tanh(a_activation) * np.tanh(a_activation)) * (prediction - y[sample_counter]) * model["W2"]
			model["b1"] = (1 - np.tanh(a_activation) * np.tanh(a_activation)) * (prediction - y[sample_counter]) * model["W2"]
			model["W2"] = (yHat[prediction] - y[sample_counter])*h_activation
			model["b2"] = (yHat[prediction] - y[sample_counter])

			sample_counter += 1

	return model

def plot_decision_boundary(pred_func, X, y):

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print(np.c_[xx.ravel(), yy.ravel()])
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    print(Z)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)


X, y = make_moons(200 , noise =0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

plt.figure(figsize =(16,32))
plt.subplot(5,2,2)
plt.title('Hidden Layer Size 1')
model = build_model(X,y,0, num_passes=100,print_loss=False)
plot_decision_boundary(lambda x: good_predict(x,model), X, y)
plt.show()
