from scipy.stats import multivariate_normal
import numpy as np
import random
from numpy import genfromtxt

np.random.seed(2)


def ten_dimensional_data():
	X = genfromtxt('data.csv', delimiter=',')
	return X

def ten_dimension_sigma():
	sigma = [ [1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,1]] 
	return sigma

def k_ten_dimensional_sigma(k):
	sigma = []
	for i in range(0,k):
		sigma.append(ten_dimension_sigma())
	return sigma

def ten_dimensional_random_mu():
	x1 = random.randint(-4,4)
	x2 = random.randint(-4,4)
	x3 = random.randint(-4,4)
	x4 = random.randint(-4,4)
	x5 = random.randint(-4,4)
	x6 = random.randint(-4,4)
	x7 = random.randint(-4,4)
	x8 = random.randint(-4,4)
	x9 = random.randint(-4,4)
	x10 = random.randint(-4,4)
	return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

def k_ten_dimensional_random_mu(k):
	mu = []
	for i in range(0,k):
		mu.append(ten_dimensional_random_mu())
	return mu



def turn_interactive_mode_on():
	plt.ion()

def random_mu():
	x1_random = random.randint(-9,9)
	y1_random = random.randint(-9,9)
	x2_random = random.randint(-9,9)
	y2_random = random.randint(-9,9)
	x3_random = random.randint(-9,9)
	y3_random = random.randint(-9,9)
	return [ [x1_random, y1_random], [x2_random, y2_random], [x3_random, y3_random]  ]

def generate_data(mu,sigma,N):
	x,y= np.random.multivariate_normal(mu, sigma, N).T
	return x,y


def experiment_1_demo_init_centroids(restart=0):
	if restart == 0:
		mu1 = [6,5]
		mu2 = [-9,6]
		mu3 = [7, -6]
		return [mu1, mu2, mu3]
	elif restart == 1:
		mu1 = [8,2]
		mu2 = [-8,9]
		mu3 = [8,-1]
		return [mu1, mu2, mu3]
	elif restart == 2:
		mu1 = [9,2]
		mu2 = [7,-6]
		mu3 = [2,8]
		return [mu1, mu2, mu3]
	elif restart == 3:
		mu1 = [8,-3]
		mu2 = [-2,-1]
		mu3 = [8,-3]
		return [mu1, mu2, mu3]
	else:
		mu1 = [0,0]
		mu2 = [0,0]
		mu3 = [0,0]
		return [mu1, mu2, mu3]


def generate_test_data_experiment_1():
	mu1 = [1,2]
	sigma1 = [[3,1],[1,2]]

	mu2 = [-1,-2]
	sigma2 = [[2,0],[0,1]]

	mu3 = [3,-3]
	sigma3 = [[1,.3],[.3,1]]

	x1,y1 = generate_data(mu1,sigma1,100)

	x2,y2 = generate_data(mu2,sigma2,100)

	x3,y3 = generate_data(mu3,sigma3,200)

	x_s = np.concatenate((x1, x2, x3))
	y_s = np.concatenate((y1, y2, y3))

	X = np.vstack((x_s,y_s))
	X = X.T

	return x1,y1,x2,y2,x3,y3,X,np.size(X,0), [mu1,mu2,mu3], [sigma1,sigma2,sigma3]

def experiment_2_demo_init_centroids(restart=0):
	if restart == 0:
		mu1 = [2,8]
		mu2 = [-7,6]
		mu3 = [-8, 2]
		return [mu1, mu2, mu3]
	elif restart == 1:
		mu1 = [5,-5]
		mu2 = [-6,-9]
		mu3 = [-4,-5]
		return [mu1, mu2, mu3]
	elif restart == 2:
		mu1 = [-2,5]
		mu2 = [-6,0]
		mu3 = [-7,6]
		return [mu1, mu2, mu3]
	elif restart == 3:
		mu1 = [2,7]
		mu2 = [3,0]
		mu3 = [2,-7]
		return [mu1, mu2, mu3]
	else:
		mu1 = [0,0]
		mu2 = [0,0]
		mu3 = [0,0]
		return [mu1, mu2, mu3]


def generate_test_data_experiment_2():
	mu1 = [0,0]
	sigma1 = [[1,-.90],[-.90,1]]

	mu2 = [-3,-3.5]
	sigma2 = [[1,.90],[.90,1]]

	mu3 = [3,2]
	sigma3 = [[1,.90],[.90,1]]

	x1,y1 = generate_data(mu1,sigma1,200)

	x2,y2 = generate_data(mu2,sigma2,200)

	x3,y3 = generate_data(mu3,sigma3,200)

	x_s = np.concatenate((x1, x2, x3))
	y_s = np.concatenate((y1, y2, y3))

	X = np.vstack((x_s,y_s))
	X = X.T

	return x1,y1,x2,y2,x3,y3,X,np.size(X,0), [mu1,mu2,mu3], [sigma1,sigma2,sigma3]

def experiment_3_demo_init_centroids(restart=0):
	if restart == 0:
		mu1 = [4,1]
		mu2 = [-5,-5]
		mu3 = [-7, 7]
		return [mu1, mu2, mu3]
	elif restart == 1:
		mu1 = [-1,-9]
		mu2 = [-3,-2]
		mu3 = [-6,9]
		return [mu1, mu2, mu3]
	elif restart == 2:
		mu1 = [-9,-3]
		mu2 = [-5,-5]
		mu3 = [-6,1]
		return [mu1, mu2, mu3]
	elif restart == 3:
		mu1 = [-6,5]
		mu2 = [-9,-7]
		mu3 = [4,5]
		return [mu1, mu2, mu3]
	else:
		mu1 = [0,0]
		mu2 = [0,0]
		mu3 = [0,0]
		return [mu1, mu2, mu3]


def generate_test_data_experiment_3():
	mu1 = [-3,3]
	sigma1 = [[1,0],[0,1]]

	mu2 = [-3,-3]
	sigma2 = [[1,0],[0,1]]

	mu3 = [4,0]
	sigma3 = [[1,0],[0,1]]

	x1,y1 = generate_data(mu1,sigma1,200)

	x2,y2 = generate_data(mu2,sigma2,200)

	x3,y3 = generate_data(mu3,sigma3,200)

	x_s = np.concatenate((x1, x2, x3))
	y_s = np.concatenate((y1, y2, y3))

	X = np.vstack((x_s,y_s))
	X = X.T

	return x1,y1,x2,y2,x3,y3,X,np.size(X,0), [mu1,mu2,mu3], [sigma1,sigma2,sigma3]
