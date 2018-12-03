import numpy as np

def compute_Z(X=np.array([[-1,-1],[-1,1],[1,-1],[1,1]]), centering=True, scaling=False):
	if centering:
		mean = np.mean(X, axis=1, keepdims=True)
		Z = X - mean
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

def project_data(Z, PCS, L, k, var=1):
	if k != 0: 
		print(Z.shape)
		Eigenvectors = PCS[:k] #first k elements
		print(Eigenvectors.T.shape)
		projected = Eigenvectors.dot(Z.T)
		return projected

	if var != 0:
		var_projected = [(i / sum(L)) for i in L]
		var_projected_array = np.cumsum(var_projected)
		print(var_projected_array)
