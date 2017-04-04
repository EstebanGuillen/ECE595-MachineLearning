import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys
import visualize as vis
import test_data
from timeit import default_timer as timer

#np.seterr(all='raise')

np.random.seed(2)


#N=600
K=3
Iterations=200
Restarts = 4

#generate data from 3 gaussians (x1,y1,x2,y2,x3,y3) and combine into one matrix X (NxD)
x1,y1,x2,y2,x3,y3,X,N,mu_true,sigma_true = test_data.generate_test_data_experiment_2()

vis.visualize_data_with_true_mu_sigma(x1,y1,x2,y2,x3,y3,mu_true,sigma_true,5.0,color_data=False,title='Training_Data_With_True_Distribution', save_plot='True')

vis.clear_plot()

vis.visualize_data_with_true_mu_sigma(x1,y1,x2,y2,x3,y3,mu_true,sigma_true,5.0,color_data=True,title='Training_Data_With_True_Distribution-Color', save_plot='True')

#print(X.shape)

cluster_zero_x = []
cluster_zero_y = []
cluster_one_x = []
cluster_one_y = []
cluster_two_x = []
cluster_two_y = []


def assign_to_cluster_lists(k,i):
		if k == 0:
			cluster_zero_x.append(X[i][0])
			cluster_zero_y.append(X[i][1])
			
		elif k == 1:
			cluster_one_x.append(X[i][0])
			cluster_one_y.append(X[i][1])
			
		elif k == 2:
			cluster_two_x.append(X[i][0])
			cluster_two_y.append(X[i][1])

def assign_to_clusters():
	
	A = np.zeros((K,N))
	total_distance_to_cluster_center = 0
	for i in range(0,N):
		arg_min = 0
		min_distance = sys.maxsize
		distance = 0
		for k in range(0,K):
			
			#distance = compute_mahalanobis_distance(X[i],centroids[k],sigma_fixed[k])
			distance = compute_euclidian_distance(X[i],centroids[k])
			if(distance < min_distance):
				arg_min = k
				min_distance = distance
		A[arg_min,i] = 1
		total_distance_to_cluster_center = total_distance_to_cluster_center + min_distance
		#this is just to visualize
		assign_to_cluster_lists(arg_min,i)
			
	return A,total_distance_to_cluster_center

def update_centroid_k(k):
	sum_aik_xi = [0,0]
	ak = 0
		
	for i in range(0,N):
		sum_aik_xi = sum_aik_xi + A[k,i]*X[i]
		ak = ak + A[k,i]
	if( ak == 0):
		#if cluster k has no membership then don't try and move it
		return mu[k]
	else:
		return sum_aik_xi/ak

	return sum_aik_xi/ak


def update_responsibility_matrix_hard_assignment():
	
	R = np.zeros((K,N))
	total_distance_to_cluster_center = 0
	for i in range(0,N):
		arg_min = 0
		min_distance = sys.maxsize
		distance = 0
		for k in range(0,K):
			
			#distance = pi[k]*compute_mahalanobis_distance(X[i],mu[k],sigma[k])
			distance = compute_mahalanobis_distance(X[i],mu[k],sigma[k])

			if(distance < min_distance):
				arg_min = k
				min_distance = distance
		R[arg_min,i] = 1
		total_distance_to_cluster_center = total_distance_to_cluster_center + min_distance
		#this is just to visualize
		assign_to_cluster_lists(arg_min,i)
			
	return R,total_distance_to_cluster_center
	#return R

def weighted_probability(xi,muk,sigmak,pik):
	D = len(muk)
	
	sigma_inverse = np.linalg.inv(sigmak)
	
	distance = (-1/2) * ((xi-muk).T.dot(sigma_inverse)).dot((xi-muk))
	det_sigmak = np.linalg.det(sigmak)
	normalizing_constant = 1 / ( ((2* np.pi)**(D/2)) * (det_sigmak**(1/2))  )
	weight = pik * float(normalizing_constant * np.exp(distance))

	return weight


def update_responsibility_matrix():
	for i in range(0,N):
		sum = 0
		arg_max = 0
		max_prob = 0
		prob = 0
		for k in range(0,K):
			prob = weighted_probability(X[i],mu[k],sigma[k],pi[k])
			R[k,i] = prob
			sum = sum + prob
			if(prob > max_prob):
				arg_max = k
				max_prob = prob
		assign_to_cluster_lists(arg_max,i)
		#normalize over the clusters, each cluster should sum to 1 
		for k in range(0,K):
			#make sure to not divide by zero, shouldn't happen but just being safe
			if(sum != 0):
				R[k,i] = R[k,i]/sum
			else:
				#print('update responsibility, sum zero')
				R[k,i] = 0
	
	'''
	#normalize over the data
	for k in range(0,K):
		sum_k = 0
		for i in range(0,N):
			sum_k = sum_k + R[k,i]
		R[k,i] = R[k,i]/sum_k
	#return R_normalized'''
	return R

def update_mu_k(k):
	sum_rik_xi = [0,0]
	rk = 0
		
	for i in range(0,N):
		sum_rik_xi = sum_rik_xi + R[k,i]*X[i]
		rk = rk + R[k,i]
	if rk == 0:
		#print('update_mu_k - rk zero')
		#if cluster k has no membership then don't try and move it
		return mu[k]
	else:
		return sum_rik_xi/rk

def update_sigma_k(k):
	small_diagonal = 1e-1 * np.eye(2,dtype=float)
	sum_rik_xi_xi_T = small_diagonal
	rk = 0
	for i in range(0,N):
		sum_rik_xi_xi_T =  sum_rik_xi_xi_T + R[k,i]* np.outer ((X[i] - mu[k]),(X[i] - mu[k]))
		rk = rk + R[k,i]

	if rk == 0:
		#print('rk zero', rk)
		return sigma[k]
		#return small_diagonal

	sigmak = sum_rik_xi_xi_T/rk 
	#det_sigmak = np.linalg.det(sigmak)
	#inv_sigmak = np.linalg.inv(sigmak)

	
	#if (det_sigmak < 1e-9 ):	
		#return sigma[k]

	return sigmak 	

def update_pi_k(k):
	rk = 0
	for i in range(0,N):
		rk = rk + R[k,i]
	return rk/N

def compute_log_likelihood():
	sum = 0
	for i in range(0,N):
		inner_sum = 0
		for k in range(0,K):
			inner_sum = inner_sum + weighted_probability(X[i],mu[k],sigma[k],pi[k])
		sum = sum + np.log(inner_sum)
	return sum

def compute_mahalanobis_distance(xi,muk,sigmak):
	sigmak_inv = np.linalg.inv(sigmak)
	d = np.sqrt(np.dot(np.dot(np.transpose((xi-muk) ),sigmak_inv), (xi-muk) ))
	return d

def compute_euclidian_distance(xi,muk):
	d = [(a - b)**2 for a, b in zip(xi, muk)]
	d = math.sqrt(sum(d))
	return d

plt.clf()
plt.ion()
	

print('')
max_log_likelihood = 0
for restarts in range(0,Restarts):
	print('')
	
	plt.clf()


	initial_mu = test_data.random_mu()

	#initialize responsibility matrix KxN
	R = np.zeros((K,N), np.float64)

	#initialize gaussian means
	mu = np.array(list(initial_mu), np.float64)
	#print(mu.shape)

	#initialize gaussian covariances
	#sigma = np.array([ [ [1,0],[0,1] ],[ [1,0],[0,1] ],[ [1,0],[0,1] ] ], np.float64)
	sigma = np.array([ [ [5,0],[0,5] ],[ [5,0],[0,5] ],[ [5,0],[0,5] ] ], np.float64)

	#print('init mu', mu);
	#print('init sigma', sigma)

	#initialize membership weights
	pi = np.array([1/K,1/K,1/K], np.float64)

	#initialize loglikelihood array
	#L = np.zeros(Iterations,)
	L = []

	vis.visualize_gmm(x1,y1,x2,y2,x3,y3,mu,sigma,5.0,iter='init',restart=restarts, save_plot='True', title='GMM_Using_EM_Algorithm (Max Log Likelihood Exit Criteria)')
	total_iterations_em = 0
	max_log_likelihood = compute_log_likelihood()
	log_likelihood = compute_log_likelihood()
	#EM algorithm for GMM
	start = timer()
	for l in range(0,Iterations):
		plt.clf()
		#Reset cluster assignments
		cluster_zero_x = []
		cluster_zero_y = []
		cluster_one_x = []
		cluster_one_y = []
		cluster_two_x = []
		cluster_two_y = []

		#E-step
		#update responsibility matrix R -> KxN
		R = update_responsibility_matrix()

	
		#M-step
		for k in range(0,K):
			mu[k] = update_mu_k(k)
			sigma[k] = update_sigma_k(k)
			pi[k] = update_pi_k(k)
		
		
		log_likelihood = compute_log_likelihood()
		if (log_likelihood > max_log_likelihood):
			max_log_likelihood = log_likelihood
		L.append(log_likelihood)

		total_iterations_em = total_iterations_em + 1
		vis.visualize_gmm(x1,y1,x2,y2,x3,y3,mu,sigma,0.1,iter=l,restart=restarts, title='GMM_Using_EM_Algorithm (Max Log Likelihood Exit Criteria)')
		#vis.visualize_kmeans_via_em(cluster_zero_x,cluster_zero_y,cluster_one_x,cluster_one_y,cluster_two_x,cluster_two_y,mu,sigma,.1,color_data=True)


		#if ( (l > 0) and (L[l] == L[l-1]) ) :
		if ( (l>0) and (math.isclose(L[l],L[l-1],rel_tol=1e-9) ) ):
			break
	end = timer()

	distribution = [len(cluster_zero_x)/N,len(cluster_one_x)/N,len(cluster_two_x)/N ]
	print('iterations:', total_iterations_em, 'max_log_likelihood:', "%.15f" % max_log_likelihood, 'time:', end-start,'distribution', distribution)
	
	vis.clear_plot()
	vis.visualize_gmm(cluster_zero_x,cluster_zero_y,cluster_one_x,cluster_one_y,cluster_two_x,cluster_two_y,mu,sigma,5.0,color_data=True, iter='final', restart=restarts, save_plot='True',title='GMM_Using_EM_Algorithm (Max Log Likelihood Exit Criteria)')
	#vis.visualize_gmm(x1,y1,x2,y2,x3,y3,mu,sigma,5.0)


	
	vis.clear_plot()
	
	vis.visualize_log_likelihood(L,restart=restarts, save_plot='True',title='Log_Likelihood_At_Each_Iteration (Max Log Likelihood Exit Criteria)')
	
	#vis.pause(5.0)
	

	plt.clf()



	#initialize responsibility matrix KxN
	R = np.zeros((K,N), np.float64)

	#initialize gaussian means
	mu = np.array(list(initial_mu), np.float64)
	#print(mu.shape)

	#initialize gaussian covariances
	#sigma = np.array([ [ [1,0],[0,1] ],[ [1,0],[0,1] ],[ [1,0],[0,1] ] ], np.float64)
	sigma = np.array([ [ [5,0],[0,5] ],[ [5,0],[0,5] ],[ [5,0],[0,5] ] ], np.float64)

	#initialize membership weights
	pi = np.array([1/K,1/K,1/K], np.float64)

	#initialize loglikelihood array
	#L = np.zeros(Iterations,)
	L = []
	mu_list = []
	sigma_list = []

	vis.visualize_gmm(x1,y1,x2,y2,x3,y3,mu,sigma,5.0,iter='init',restart=restarts, save_plot='True',title='GMM_Using_EM_Algorithm (Stable mu, Sigma Exit Criteria)')
	total_iterations_em = 0
	max_log_likelihood = compute_log_likelihood()
	log_likelihood = compute_log_likelihood()
	#EM algorithm for GMM
	start = timer()
	for l in range(0,Iterations):
		plt.clf()
		#Reset cluster assignments
		cluster_zero_x = []
		cluster_zero_y = []
		cluster_one_x = []
		cluster_one_y = []
		cluster_two_x = []
		cluster_two_y = []

		#E-step
		#update responsibility matrix R -> KxN
		R = update_responsibility_matrix()

	
		#M-step
		for k in range(0,K):
			mu[k] = update_mu_k(k)
			sigma[k] = update_sigma_k(k)
			pi[k] = update_pi_k(k)
		
		mu_list.append(np.matrix.copy(mu))
		sigma_list.append(np.matrix.copy(sigma))

		log_likelihood = compute_log_likelihood()
		if (log_likelihood > max_log_likelihood):
			max_log_likelihood = log_likelihood
		L.append(log_likelihood)

		total_iterations_em = total_iterations_em + 1
		vis.visualize_gmm(x1,y1,x2,y2,x3,y3,mu,sigma,0.1,iter=l,restart=restarts,title='GMM_Using_EM_Algorithm (Stable mu, Sigma Exit Criteria)')
		#vis.visualize_kmeans_via_em(cluster_zero_x,cluster_zero_y,cluster_one_x,cluster_one_y,cluster_two_x,cluster_two_y,mu,sigma,.1,color_data=True)


		#if ( (l > 0) and (L[l] == L[l-1]) ) :
		#if ( (l>0) and (math.isclose(L[l],L[l-1],rel_tol=1e-9) ) ):
			#break

		'''if(total_iterations_em > 3
			and np.array_equal(mu_list[l],mu_list[l-1]) 
			and np.array_equal(sigma_list[l],sigma_list[l-1]) ):
			print ('mu', mu_list[l])
			print ('mu t-1', mu_list[l-1])
			print ('sigma', sigma_list[l])
			print ('sigma t-1', mu_list[l-1])
			#print ('mu t-2', mu_list[l-2])
			
			break'''
		
		if(total_iterations_em > 3
			and np.allclose(mu_list[l],mu_list[l-1],rtol=1e-09) 
			and np.allclose(sigma_list[l],sigma_list[l-1],rtol=1e-09) ):
			
			break
	end = timer()

	distribution = [len(cluster_zero_x)/N,len(cluster_one_x)/N,len(cluster_two_x)/N ]
	print('iterations:', total_iterations_em, 'max_log_likelihood:', "%.15f" % max_log_likelihood, 'time:', end-start,'distribution', distribution)
	
	vis.clear_plot()
	vis.visualize_gmm(cluster_zero_x,cluster_zero_y,cluster_one_x,cluster_one_y,cluster_two_x,cluster_two_y,mu,sigma,5.0,color_data=True, iter='final', restart=restarts, save_plot='True',title='GMM_Using_EM_Algorithm (Stable mu, Sigma Exit Criteria)')
	#vis.visualize_gmm(x1,y1,x2,y2,x3,y3,mu,sigma,5.0)


	
	vis.clear_plot()
	
	vis.visualize_log_likelihood(L,restart=restarts, save_plot='True', title='Log_Likelihood_At_Each_Iteration (Stable mu, Sigma Exit Criteria)')
	
	#vis.pause(5.0)
	



	plt.clf()

