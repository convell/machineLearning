import os
import pca
import numpy as np
import matplotlib.pyplot as plt


def load_data(input_dir):
	files = os.listdir(input_dir)
	input_array = []
	for file in files:
		input_array.append(plt.imread(input_dir+file).flatten())

	numpy_data = np.array(input_array).T
	numpy_data.astype(float)
	return numpy_data


def compress_images(DATA,k):

	Z = pca.compute_Z(DATA)
	cov = pca.compute_covariance_matrix(Z)
	PCS = pca.find_pcs(cov)
	L, PCS_1 = PCS
	projected_data = pca.project_data(Z, PCS_1, L, k, 0)
	print(projected_data.shape)
	PCS_1=PCS_1[:k]
	print(PCS_1.shape)
	compressed = (projected_data.T).dot(PCS_1)

	os.makedirs(os.getcwd()+"/Output", exist_ok=True)
	output_dir = os.getcwd()+"/Output"
	
	for image in compressed.T:
		image = 255*(image - np.min(image))/np.ptp(image)
		image = np.reshape(image, (60,48))


		plt.imsave(output_dir+'/'+str(number_counter)+".png", image, cmap='gray')



cwd = os.getcwd() +"/Data/Train/"
data = load_data(cwd)
compress_images(data,153)


#Z_2= pca.compute_Z()


#cov_2 =pca.compute_covariance_matrix(Z_2)

#PCS_2 = pca.find_pcs(cov_2)

#L2, PCS_2 = PCS_2

#pca.project_data(Z_2, PCS_2, L2, 0, 1)
