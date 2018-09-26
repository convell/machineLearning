import numpy as np
import random
import math

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



	##print(Intial_Cluster_Centers)

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
			##print(current_x, center)
			for index in range(Dimensions):
				distance_running += (current_x[index] - center[index])**2

			distance_array[center_index] = math.sqrt(distance_running)

			center_index += 1

		min_distance_index = np.where(distance_array == distance_array.min())[0][0]
		Cluster_Centers[min_distance_index].append(list(current_x))

	#for k_index in range(K):
	#	#print(Cluster_Centers[k_index])

	return np.array(Cluster_Centers)


			#return index of minimum distance to assign it to that cluster






	#	for item in X:
	#		Dimension_Total += item[index]
	#	Dimension_Total_Mean = Dimension_Total_Mean/X.shape[0]

	#	Cluster_Center[index] = Dimension_Total_Mean/K
