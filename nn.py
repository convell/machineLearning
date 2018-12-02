import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def softmax(x):
	r=np.exp(x - np.max(x))
	return r/np.sum(r,axis =1, keepdims = True)


def calculate_loss(X, y, model={"W1":np.array([1,2]),"W2":np.array([2,3]),"b1":4,"b2":4}):
	loss = 0
	sample_counter = 0
	for sample in X: #for each sample
		yHat = loss_predict(sample, model) # get a prediction
		loss += np.log(yHat[0][y[sample_counter]]) #log the prediction of the right class
		sample_counter += 1

	loss = -(1/sample_counter) * loss #plug into the rest of our loss function

	return loss

def loss_predict(x, model): #same as normal predict but returns yhat
	a_activation = x.dot(model["W1"]) + model["b1"]
	h_activation = np.tanh(a_activation)
	z_activation = h_activation.dot(model["W2"]) + model["b2"]
	yHat = softmax(z_activation)
	return yHat

def good_predict(x, model): #returns index of highest probable class
	a_activation = x.dot(model["W1"]) + model["b1"]
	h_activation = np.tanh(a_activation)
	z_activation = h_activation.dot(model["W2"]) + model["b2"]
	yHat = softmax(z_activation)
	return np.argmax(yHat, axis=1)


def build_model(X,y,nn_hdim, num_passes=200000,print_loss=False): 
	#follows normal NN steps:
	#Intialized weights (looked up a way to generate okay values for tanh activation)
	#Forward Pass
	#Backprop
	#Gradient Descent/weight updates
	#Repeat

	#Intialize shape of weights
	model = {"W1":np.zeros((2,nn_hdim)),"W2":np.zeros((nn_hdim,2)),"b1":np.zeros((1,nn_hdim)),"b2":np.zeros((1,2))}
	learning_rate = 0.01
	#initialize weights for tanh activation
	model["W1"] = np.random.randn(2,nn_hdim) * np.sqrt(1/nn_hdim)
	model["W2"] = np.random.randn(nn_hdim,2)* np.sqrt(1/2)


	for epoch in range(num_passes):
		#forward pass
		a_activation = X.dot(model["W1"]) + model["b1"]
		h_activation = np.tanh(a_activation)
		z_activation = h_activation.dot(model["W2"]) + model["b2"]
		yHat = softmax(z_activation)
		sample_counter = 0

		for sample in yHat: #couldnt figure a linear algebra way of doing it
			sample[y[sample_counter]] -= 1
			yHat[sample_counter] = sample
			sample_counter += 1


		oldW2 = model["W2"]

		#translated the derivatives given to us in project description into real values
		back_prop_w2 = ((h_activation.T).dot(yHat))
		back_prop_b2 = np.sum(yHat, axis=0, keepdims=True)
		back_prop_w1 = np.dot(X.T, (yHat.dot(oldW2.T)) * (1 - (h_activation*h_activation)))
		back_prop_b1 = np.sum((yHat.dot(oldW2.T)) * (1 - (h_activation*h_activation)), axis=0)

		#gradient descent formula
		model["W2"] = model["W2"] - (back_prop_w2 * learning_rate)
		model["b2"] = model["b2"] - (back_prop_b2 * learning_rate)
		model["W1"] = model["W1"] - (back_prop_w1 * learning_rate)
		model["b1"] = model["b1"] - (back_prop_b1 * learning_rate)

		if print_loss == True:
			if epoch % 1000 == 0:
				print(calculate_loss(X,y,model))


	return model


def plot_decision_boundary(pred_func, X, y):

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)



#main
np.random.seed(0)
X, y = make_moons(200 , noise =0.20)

plt.figure(figsize =(16,36))
hidden_layer_dimensions = [1, 2, 3, 4]
count = 1

for dim in hidden_layer_dimensions:
	plt.subplot(3,2,count)
	plt.title('Hidden Layer Size %d' % dim)
	model = build_model(X, y, dim, num_passes=20000, print_loss=False)
	print(calculate_loss(X,y,model))
	plot_decision_boundary(lambda x: good_predict(x,model), X, y)
	count += 1

plt.show()