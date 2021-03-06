import os
import pca
import numpy as np
import matplotlib.pyplot as plt


def load_data(input_dir):
	files = os.listdir(input_dir)
	files.sort(key = lambda x: int(x[:5]))
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
	PCS_1=PCS_1.T
	PCS_1 = PCS_1[:k]
	print(PCS_1.shape)
	compressed = (projected_data).dot(PCS_1)

	os.makedirs(os.getcwd()+"/Output", exist_ok=True)
	output_dir = os.getcwd()+"/Output"
	number_counter = 0
	for image in compressed.T:
		image = 255*(image - np.min(image))/np.ptp(image)
		image = np.reshape(image, (60,48))


		plt.imsave(output_dir+'/'+str(number_counter)+".png", image, cmap='gray')
		number_counter += 1



cwd = os.getcwd() +"/Data/Train/"
data = load_data(cwd)
compress_images(data,2)


Z_2= pca.compute_Z()


cov_2 =pca.compute_covariance_matrix(Z_2)
print("cov" ,cov_2)

PCS_2 = pca.find_pcs(cov_2)

L2, PCS_2 = PCS_2

print(pca.project_data(Z_2, PCS_2, L2, 0, 0.3))
