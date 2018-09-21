import numpy as np
import math

def main():

	#K-NN Data
	X_train = np.array([[1,5],[2,6],[2,7],[3,7],[3,8],[4,8],[5,1],[5,9],[6,2],[7,2],[7,3],[8,3],[8,4],[9,5]])
	Y_train = np.array([[-1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[1]])
	X_test = np.array([[1,1], [2,1], [0,10], [10,10], [5,5], [3,10], [9,4], [6,2], [2,2], [8,7]])
	Y_test = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])
	K = 5
	
	Accuracy = KNN_test(X_train,Y_train,X_test,Y_test,K)
	print("KNN Accuracy: ", Accuracy, "%")


# DT section
#DT_train_binary(X,Y,max_depth):


#DT_test_binary(X,Y,DT):

#K-NN
def KNN_test(X_train,Y_train,X_test,Y_test,K):
	index_of_test = 0
	accurate_label = 0
	for current_x_test in X_test:
		index_of_training = 0
		distance_array = np.zeros((X_train.shape[0],2))
		label_value = 0	
		for current_x_train in X_train:
			distance_running = 0
			for index in range(np.ndim(X_test)):
				distance_running += (current_x_test[index] - current_x_train[index])**2
			
			distance = math.sqrt(distance_running)
			
			distance_array[index_of_training][0] = distance
			distance_array[index_of_training][1] = index_of_training
			#print("Test Point:", current_x_test, "Train Point:", current_x_train, "Distance:", distance)

			index_of_training += 1

		distance_array = distance_array[distance_array[:,0].argsort()]

		for x in range(K):
			index_for_training_label = int(distance_array[x][1])
			#print(Y_train[index_for_training_label])
			label_value += Y_train[index_for_training_label][0]

		print("Test Point:", current_x_test ,", Classified as:",label_value, ", K Value of:", K, ", Actual Label:",Y_test[index_of_test][0])
		if label_value == Y_test[index_of_test][0]:
			accurate_label += 1

		#print(distance_array)
		index_of_test += 1

	Accuracy = (accurate_label/index_of_test)*100

	return Accuracy




if __name__ == "__main__":
    main()