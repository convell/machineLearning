import numpy as np
import random
import math

def main():

	#K-NN Data
	X_train = np.array([[1,5],[2,6],[2,7],[3,7],[3,8],[4,8],[5,1],[5,9],[6,2],[7,2],[7,3],[8,3],[8,4],[9,5]])
	Y_train = np.array([[-1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[1]])
	X_test = np.array([[1,1], [2,1], [0,10], [10,10], [5,5], [3,10], [9,4], [6,2], [2,2], [8,7]])
	Y_test = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])
	K = 5

	#Accuracy = KNN_test(X_train,Y_train,X_test,Y_test,K)
	#print("KNN Accuracy: ", Accuracy, "%")

	#K_Means
	X = np.array([[1,0],[7,4],[9,6],[2,1],[4,8],[0,3],[13,5],[6,8],[7,3],[3,6],[2,1],[8,3],[10,2],[3,5],[5,1],[1,9],[10,3],[4,1],[6,6],[2,2]])
	X = np.array([[0],[1],[2],[7],[8],[9],[12],[14],[15]])
	K = 3
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

	max_depth = 4

	#DT_train_binary(X_train_2,Y_train_2,max_depth)

# DT section
def DT_train_binary(X,Y,max_depth):
	
	num_features = len(X[0])
	accuracy_array = []

	for index in range(num_features):
		accuracy_array.append(0)
	feature_counter = 0
	
	for feature_col in X.T: #For each feature, calculate accuracy
		sample_counter = 0

		for sample_feature in feature_col:
			if sample_feature == Y[sample_counter][0]:
				accuracy_array[feature_counter] = accuracy_array[feature_counter]+1
			sample_counter += 1

		accuracy_array[feature_counter] = accuracy_array[feature_counter] / sample_counter
		feature_counter += 1


	#Build DT
	if max_depth == -1:
		DT = np.zeros((max_depth,2))

	DT = np.zeros((max_depth,2))


	print(accuracy_array)




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

def K_Means(X,K):
	Old_Cluster_Center = 0
	Old_Mean = 0
	New_Mean = K_Means_Intial_Cluster(X,K)
	Total_Num_Points = 1
	print("Intial Cluster Centers: ")
	for row in New_Mean:
		print(row)

	for dimension_length in X.shape:
		Total_Num_Points = Total_Num_Points * dimension_length
	while True:
		Clusters = Assign_Data(New_Mean,X,K)
		New_Mean = Mean_Calculator(Clusters,Total_Num_Points)
		print()
		print("Clusters:")
		for Cluster_Group in list(Clusters):
			print ("Cluster Group ",list(Clusters).index(Cluster_Group), ": " ,Cluster_Group)
		if np.array_equal(Old_Mean, New_Mean):
			break
		Old_Mean = New_Mean



	#print(Intial_Cluster_Centers)

def Mean_Calculator(Clusters, Total_Num_Points):
	Transposed_Clusters = np.transpose(Clusters)
	dims = len(Clusters[0][0])
	Cluster_Centers = np.zeros((Clusters.shape[0],dims))
	Total = 0
	for dimension in range(dims):
		Cluster_Group_Counter = 0
		for Cluster_group in Clusters:
			for Point in Cluster_group:
				Total += Point[dimension]

			Mean = Total / Total_Num_Points
			Cluster_Centers[Cluster_Group_Counter][dimension] = Mean
			Cluster_Group_Counter += 1

	return Cluster_Centers
	#for index in range(np.ndim(Clusters)):



		
def K_Means_Intial_Cluster(X,K):
	#Gets equidistance points on each axis (max int on axis divided by K) and shuffles them in plac
	Cluster_Centers = np.zeros((K,X.shape[-1]))
	Dimension_Total = 0
	Transposed_X = np.transpose(X)
	Dimensions = X.shape[-1]
	for index in range(Dimensions):
		Feature_Max = max(Transposed_X[index])
		Even_Distance_Values = Feature_Max/K

		for k_index in range(K):
			Cluster_Centers[k_index][index] = Feature_Max - (Even_Distance_Values * k_index)
	
	for i in range(Cluster_Centers.shape[1]):
		np.random.shuffle(Cluster_Centers[:,i])

	return Cluster_Centers

def Assign_Data(Intial_Cluster_Centers,X,K):
	Cluster_Centers = []
	Dimensions = X.shape[-1]

	for k_index in range(K):
		Cluster_Centers.append([]) #create K columns of arrays 

	for current_x in X:
		distance_array = np.zeros((Intial_Cluster_Centers.shape[0]))
		center_index = 0

		for center in Intial_Cluster_Centers:
			distance_running = 0
			#print(current_x, center)
			for index in range(Dimensions):
				distance_running += (current_x[index] - center[index])**2

			distance_array[center_index] = math.sqrt(distance_running)

			center_index += 1

		min_distance_index = np.where(distance_array == distance_array.min())[0][0]
		Cluster_Centers[min_distance_index].append(list(current_x))

	#for k_index in range(K):
	#	print(Cluster_Centers[k_index])

	return np.array(Cluster_Centers)


			#return index of minimum distance to assign it to that cluster






	#	for item in X:
	#		Dimension_Total += item[index]
	#	Dimension_Total_Mean = Dimension_Total_Mean/X.shape[0]

	#	Cluster_Center[index] = Dimension_Total_Mean/K






if __name__ == "__main__":
    main()