import numpy as np

def compute_Z(X=np.array([[-1,-2],[-2,1],[4,-1],[1,1]]), centering=True, scaling=False):
	if centering:
		mean = np.mean(X, axis=0, keepdims=True)
		print(mean)
		print(X)
		Z = (X - mean)
		print(Z)
		if scaling:
			std = np.std(X, axis=1, keepdims=True)
			Z = np.divide(X, std, where=std!=0)

	elif scaling:
		std = np.std(X, axis=1, keepdims=True)
		Z = np.divide(X, std, where=std!=0)

	return(Z)

def compute_covariance_matrix(Z):
	return (Z.T).dot(Z)

def find_pcs(COV):
	return np.linalg.eig(COV)

def project_data(Z, PCS, L, k, var):
	if k != 0: 
		print(Z.shape)
		Eigenvectors = PCS.T #first k elements
		Eigenvectors = Eigenvectors[:k]
		print("eigenvectors",Eigenvectors.T.shape)
		projected = Z.dot(Eigenvectors.T)
		print("projected",projected.shape)
		return projected

	if var != 0:
		var_projected = [(i / sum(L)) for i in L]
		var_projected_array = np.cumsum(var_projected)
		k = 1
		print(var_projected_array)
		for variance in var_projected_array:
			if var < variance:
				print(variance)
				Eigenvectors = PCS.T
				Eigenvectors = Eigenvectors[:k]
				projected = Z.dot(Eigenvectors.T)
				return projected
			k += 1
