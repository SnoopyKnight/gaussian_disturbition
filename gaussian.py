import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky


def zero_mean(dim):
	mean = np.zeros([dim])
	return mean


def covariance_matrix(p, dim):
	cov = np.zeros([dim, dim])	
	for i in range(dim):
		cnt = 0
		for j in range(i,dim):
			if(i==j):
				cov[i][j] = 1			
			else:
				cov[i][j] = pow(p,cnt+1)
				cnt = cnt + 1
	cov = cov + cov.T - np.diag(cov.diagonal())             
	return cov


def gauss_distribution(mean, cov, sampleNo, dim):
	[eigenvalues, eigenvectors] = np.linalg.eig(cov)
	# l = np.matrix(np.diag(np.sqrt(eigenvalues)))
	# print(l)
	R = cholesky(cov)
	s = np.dot(np.random.randn(sampleNo, dim), R) + mean
	return s


def node_plot(s):
	plt.plot()
	plt.plot(s[:,0],s[:,1],'+')
	plt.show()


def histgram(s):
	plt.plot()
	plt.hist(s[:,1], bins='auto')
	plt.show()


def main():
	p1 = 0.9
	p2 = 0.5
	sampleNo = 100
	dim = 20

	mean = zero_mean(dim)
	#cov = covariance_matrix(p1, dim)
	cov = covariance_matrix(p2, dim)
	print(cov)
	s = gauss_distribution(mean, cov, sampleNo, dim)
	node_plot(s)
	# histgram(s)

if __name__ == '__main__':
	main()