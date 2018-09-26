import numpy as np
import random
import math
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.left_features_used = []
        self.right_features_used = []
        self.left_points = []
        self.right_points = []
        self.left_accuracy = 0
        self.right_accuracy = 0
        self.feature_index = None
        self.accuracy = None

# DT section
def DT_train_binary(X,Y,max_depth):
	
	num_features = len(X[0])
	accuracy_array = []
	DT_array = []
	feature_used_array = []
	DT = Tree()
	highest_accuracy = 0

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

	#Build DT Root
	for result in accuracy_array:

		if result == 1.0:
			DT.feature_index = accuracy_array.index(result)
			DT.left = 0
			DT.right = 1
			DT.accuracy = result
			return DT

		elif result == 0.0:
			DT.feature_index = accuracy_array.index(result)
			DT.left = 1
			DT.right = 0
			DT.accuracy = result
			return DT

		else:
			if result > highest_accuracy:
				highest_accuracy = result

	DT.feature_index = accuracy_array.index(highest_accuracy)
	DT.left = 0
	DT.right = 1
	DT.accuracy = highest_accuracy
	sample_counter = 0
	accuracy_counter_left = 0
	accuracy_counter_right = 0

	for sample in X:
		if sample[DT.feature_index] == Y[sample_counter]:
			if Y[sample_counter] == DT.left:
				accuracy_counter_left += 1
			else:
				accuracy_counter_right += 1
		if sample[DT.feature_index] == DT.left:
			sample = list(sample)
			sample.append(sample_counter)
			sample = np.array(sample)
			DT.left_points.append(sample)
		else:
			sample = list(sample)
			sample.append(sample_counter)
			sample = np.array(sample)
			DT.right_points.append(sample)

		sample_counter += 1

	DT.left_accuracy = accuracy_counter_left / sample_counter
	DT.right_accuracy = accuracy_counter_right / sample_counter
	DT = DT_Recursion_Helper(DT,X,Y,max_depth -1)



	return DT

def DT_Recursion_Helper(DT,X,Y,max_depth):
	right_perf = False
	left_perf = False
	accuracy_array = []
	num_features = len(X[0])
	feature_counter = 0
	#print("hi", max_depth)
	for index in range(num_features):
		accuracy_array.append(0)

	if (DT.accuracy == 1.0) or (max_depth == 0):
		#print("what 6")
		return DT

	#Build left tree
	sample_array = np.array(DT.left_points).T[-1:]
	left_points_transpose = np.array(DT.left_points).T[:-1]
	for feature_col in left_points_transpose: #For each feature, calculate accuracy
		sample_counter = 0
		for sample_feature in feature_col:
			if sample_feature == Y[sample_array[0][sample_counter]]:
				accuracy_array[feature_counter] = accuracy_array[feature_counter]+1
			sample_counter += 1

		accuracy_array[feature_counter] = accuracy_array[feature_counter] / sample_counter
		feature_counter += 1

	highest_accuracy = DT.left_accuracy

	for result in accuracy_array:
		if result == 1.0:
			DT.left = Tree()
			DT.left.feature_index = accuracy_array.index(result)
			DT.left.left = 0
			DT.left.right = 1
			DT.left.accuracy = result
			#print("what 5")
			left_perf = True

		elif result == 0.0:
			DT.left = Tree()
			DT.left.feature_index = accuracy_array.index(result)
			DT.left.left = 1
			DT.left.right = 0
			DT.left.accuracy = result
			#print("what 4")
			left_perf = True
		else:
			if result > highest_accuracy:
				highest_accuracy = result

		if highest_accuracy != DT.left_accuracy:
			DT.left = Tree()
			DT.left.feature_index = accuracy_array.index(highest_accuracy)
			DT.left.left = 0
			DT.left.right = 1
			DT.left.accuracy = highest_accuracy
			sample_counter = 0
			accuracy_counter_left = 0
			accuracy_counter_right = 0


	accuracy_array = []
	num_features = len(X[0])
	feature_counter = 0

	for index in range(num_features):
		accuracy_array.append(0)

	if (DT.accuracy == 1.0):
		#print("what 3")
		return DT

	#build right tree
	sample_array = np.array(DT.right_points).T[-1:]
	right_points_transpose = np.array(DT.right_points).T[:-1]

	for feature_col in right_points_transpose: #For each feature, calculate accuracy
		sample_counter = 0
		
		for sample_feature in feature_col:
			if sample_feature == Y[sample_array[0][sample_counter]]:
				accuracy_array[feature_counter] = accuracy_array[feature_counter]+1
			sample_counter += 1

		accuracy_array[feature_counter] = accuracy_array[feature_counter] / sample_counter
		feature_counter += 1

	highest_accuracy = DT.left_accuracy

	for result in accuracy_array:
		if result == 1.0:
			DT.right = Tree()
			DT.right.feature_index = accuracy_array.index(result)
			DT.right.left = 0
			DT.right.right = 1
			DT.right.accuracy = result
			#print("what 2")
			right_perf = True

		elif result == 0.0:
			DT.right = Tree()
			DT.right.feature_index = accuracy_array.index(result)
			DT.right.left = 1
			DT.right.right = 0
			DT.right.accuracy = result
			right_perf = True
		else:
			if result > highest_accuracy:
				highest_accuracy = result

	if highest_accuracy != DT.left_accuracy:
		DT.right = Tree()
		DT.right.feature_index = accuracy_array.index(highest_accuracy)
		DT.right.left = 0
		DT.right.right = 1
		DT.right.accuracy = highest_accuracy
		sample_counter = 0
		accuracy_counter_left = 0
		accuracy_counter_right = 0
		DT.left_points = []
		DT.right_points = []

		for sample in X:
			if sample[DT.feature_index] == Y[sample_counter]:
				if Y[sample_counter] == DT.left:
					accuracy_counter_left += 1
				else:
					DT.left_points.append(sample)
			else:
				DT.right_points.append(sample)

			sample_counter += 1

	if not left_perf == True:
		DT.left = DT_Recursion_Helper(DT.left,X,Y,max_depth-1)

	if not right_perf == True:
		DT.right = DT_Recursion_Helper(DT.left,X,Y,max_depth-1)
	DT.accuracy = DT.left_accuracy + DT.right_accuracy
	return DT


def DT_train_binary_best(X_train, Y_train, X_val, Y_val):

	Trained_DT = DT_train_binary(X_train,Y_train,-1)

	sample_counter = 0
	accuracy_array = []
	
	print("Original DT Accuracy",DT_test_binary(X_train,Y_train,Trained_DT))
	#print("Val", DT_test_binary(X_val,Y_val,Trained_DT,best_flag = False))
	#print("Val left", DT_test_binary(X_val,Y_val,Trained_DT.left,best_flag = False))

	New_DT = DT_train_binary_best_helper(X_val, Y_val, Trained_DT, accuracy_array)
	print("Best DT Accuracy",DT_test_binary(X_val,Y_val,New_DT))


	return New_DT

def DT_train_binary_best_helper(X, Y, DT, accuracy_array,counter = 0):
	temp_left = []
	temp_right = []

	accuracy_array.append([])
	accuracy_array[counter].append(str(DT_test_binary(X,Y,DT))+"TOP")

	if not type(DT.left) == int:
		accuracy_array[counter].append(DT_test_binary(X,Y,DT.left))
		if DT_test_binary(X,Y,DT.left) < .5:
			DT.left.left = 0
			DT.right.right = 0
			#print( "NEW ACC", DT_test_binary(X,Y,DT))
		temp_left.append(DT_train_binary_best_helper(X, Y, DT.left, accuracy_array , counter + 1))
	else:
		accuracy_array[counter].append((DT_test_binary(X,Y,DT)))
	

	if not type(DT.right) == int:
		if DT_test_binary(X,Y,DT.right) < .5:
			DT.right.left = 0
			DT.right.right = 1
		accuracy_array[counter].append(DT_test_binary(X,Y,DT.right))
		temp_right.append(DT_train_binary_best_helper(X, Y, DT.right, accuracy_array , counter + 1))
	else:
		accuracy_array[counter].append(DT_test_binary(X,Y,DT))
		

	temp = {"left":temp_left,"right":temp_right}
	return(DT)




def Print_DT(DT):
	#print("Feature:", DT.feature_index, "Left:")
	if not type(DT.left) == int:
		#print("  ")
		Print_DT(DT.left)

	if not type(DT.right) == int:
		#print("  ")
		Print_DT(DT.right)


def Clear_DT_Points(DT):
	
	if not type(DT.left) == int:
		DT.left_points = []
		Clear_DT_Points(DT.left)
	if not type(DT.right) == int:
		DT.right_points = []	
		Clear_DT_Points(DT.right)


def DT_test_binary(X,Y,DT,best_flag = False):


	accuracy_counter = 0
	sample_counter = 0
	Clear_DT_Points(DT)
	for sample in X:
		
		if Y[sample_counter] == DT_test_binary_helper(sample,DT):
			accuracy_counter += 1
		sample_counter += 1

	accuracy = accuracy_counter/sample_counter
	#print("Accuracy: ", accuracy)
	return accuracy


def DT_test_binary_helper(sample,DT):

	accuracy_counter = 0
	if sample[DT.feature_index] == 0:
		if type(DT.left) == int:
			DT.left_points.append(sample)
			return DT.left
		else:
			DT.left_points.append(sample)
			return DT_test_binary_helper(sample,DT.left)
	else:
		if type(DT.right) == int:
			DT.right_points.append(sample)
			return DT.right
		else:
			DT.right_points.append(sample)
			return DT_test_binary_helper(sample,DT.right)
