import numpy as np
import random
import math
from kmeans import K_Means
from knn import KNN_test
from decisiontrees import DT_train_binary, DT_test_binary, DT_train_binary_best

def main():

	#K-NN Data
	X_train = np.array([[1,5],[2,6],[2,7],[3,7],[3,8],[4,8],[5,1],[5,9],[6,2],[7,2],[7,3],[8,3],[8,4],[9,5]])
	Y_train = np.array([[-1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[1]])
	X_test = np.array([[1,1], [2,1], [0,10], [10,10], [5,5], [3,10], [9,4], [6,2], [2,2], [8,7]])
	Y_test = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])
	K = 1

	Accuracy = KNN_test(X_train,Y_train,X_test,Y_test,K)
	print("KNN Accuracy: ", Accuracy, "%")

	#K_Means
	X = np.array([[1,0],[7,4],[9,6],[2,1],[4,8],[0,3],[13,5],[6,8],[7,3],[3,6],[2,1],[8,3],[10,2],[3,5],[5,1],[1,9],[10,3],[4,1],[6,6],[2,2]])
	#X = np.array([[0],[1],[2],[7],[8],[9],[12],[14],[15]])
	K = 2
	K_Means(X,K)

	# Training Set 1:
	X_train_1 = np.array([[0,1], [0,0], [1,0], [0,0], [1,1]])
	Y_train_1 = np.array([[1], [0], [0], [0], [1]])
	# Validation Set 1:
	X_val_1 = np.array([[0,0], [0,1], [1,0], [1,1]])
	Y_val_1 = np.array([[0], [1], [0], [1]])
	# Testing Set 1:
	X_test_1 = np.array([[0,0], [0,1], [1,0], [1,1]])
	Y_test_1 = np.array([[1], [1], [0], [1]])
	# Training Set 2:
	X_train_2 = np.array([[0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1], [1,1,0,1], [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,0,0]])
	Y_train_2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])
	# Validation Set 2:
	X_val_2 = np.array([[1,0,0,0], [0,0,1,1], [1,1,0,1], [1,1,0,0], [1,0,0,1], [0,1,0,0]])
	Y_val_2 = np.array([[0], [0], [1], [0], [1], [1]])
	# Testing Set 2:
	X_test_2 = np.array([[0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1], [1,1,0,1], [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,0,0]])
	Y_test_2 = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])

	max_depth = -1

	Trained_DT = DT_train_binary(X_train_1,Y_train_1,max_depth)
	Accuracy = DT_test_binary(X_test_1,Y_test_1,Trained_DT)
	print(Accuracy)

	Trained_DT = DT_train_binary_best(X_train_1, Y_train_1, X_val_1, Y_val_1)
	Accuracy = DT_test_binary(X_test_1,Y_test_1,Trained_DT)
	print(Accuracy)

if __name__ == "__main__":
    main()
