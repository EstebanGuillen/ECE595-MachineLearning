import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys
import visualize as vis
import test_data
from timeit import default_timer as timer
import gmm
import k_means

np.random.seed(2)


#N=600
#K=7
Max_K=14
Min_K=3
Iterations=100
Restarts = 30
N=600

#generate data from 3 gaussians (x1,y1,x2,y2,x3,y3) and combine into one matrix X (NxD)
#x1,y1,x2,y2,x3,y3,X,N,mu_true,sigma_true = test_data.generate_test_data_experiment_1()
X = test_data.ten_dimensional_data()

#vis.visualize_data_with_true_mu_sigma(x1,y1,x2,y2,x3,y3,mu_true,sigma_true,1.0,color_data=False,title='Training_Data_With_True_Distribution', save_plot='True')



L_em_total = np.zeros((Restarts,Iterations), np.float64)
L_kmeans_total = np.zeros((Restarts,Iterations), np.float64)

D_kmeans_maha_total = np.zeros((Restarts,Iterations), np.float64)
D_kmeans_total = np.zeros((Restarts,Iterations), np.float64)

iterations_for_each_restart_em = np.zeros(Restarts)
iterations_for_each_restart_kmeans_maha = np.zeros(Restarts)
iterations_for_each_restart_kmeans_eucl = np.zeros(Restarts)

Optimal_K_em = np.zeros( (Restarts,(Max_K-Min_K)),np.float64 )
Optimal_K_kmeans_maha = np.zeros( (Restarts,(Max_K-Min_K)),np.float64 )
Optimal_K_kmeans_eucl = np.zeros( (Restarts,(Max_K-Min_K)),np.float64 )


for K in range(Min_K,Max_K):
	print(K)

	for restarts in range(0,Restarts):
		
		#plt.clf()


		#initial_mu = test_data.random_mu()
		initial_mu = test_data.k_ten_dimensional_random_mu(K)

		#initialize responsibility matrix KxN
		R = np.zeros((K,N), np.float64)

		#initialize gaussian means
		mu = np.array(list(initial_mu), np.float64)
		#print(mu.shape)

		#initialize gaussian covariances
		#sigma = np.array([ [ [1,0],[0,1] ],[ [1,0],[0,1] ],[ [1,0],[0,1] ] ], np.float64)
		#sigma = np.array([ [ [5,0],[0,5] ],[ [5,0],[0,5] ],[ [5,0],[0,5] ] ], np.float64)
		sigma = 2 * test_data.k_ten_dimensional_sigma(K)

		#print('init mu', mu);
		#print('init sigma', sigma)

		#initialize membership weights
		#pi = np.array([1/K,1/K,1/K], np.float64)
		pi = 1/K *np.ones(K)

		#initialize loglikelihood array
		#L = np.zeros(Iterations,)
		L = []

		#vis.visualize_gmm(x1,y1,x2,y2,x3,y3,mu,sigma,5.0,iter='init',restart=restarts, save_plot='True')
		total_iterations_em = 0
		max_log_likelihood = gmm.compute_log_likelihood(X,mu,sigma,pi,N,K)
		log_likelihood = gmm.compute_log_likelihood(X,mu,sigma,pi,N,K)
		#EM algorithm for GMM
		start = timer()
		for l in range(0,Iterations):
			#plt.clf()


			#E-step
			#update responsibility matrix R -> KxN
			R = gmm.update_responsibility_matrix(R,X,mu,sigma,pi,N,K)

		
			#M-step
			for k in range(0,K):
				
				mu[k] = gmm.update_mu_k(R,k,mu[k],N,X)
				sigma[k] = gmm.update_sigma_k(R,k,mu,sigma[k],N,X)
				pi[k] = gmm.update_pi_k(R,k,N)
			
			
			log_likelihood = gmm.compute_log_likelihood(X,mu,sigma,pi,N,K)
			L_em_total[restarts,l] = log_likelihood
			if (log_likelihood > max_log_likelihood):
				max_log_likelihood = log_likelihood
			L.append(log_likelihood)

			total_iterations_em = total_iterations_em + 1
			#vis.visualize_gmm(x1,y1,x2,y2,x3,y3,mu,sigma,0.1,iter=l,restart=restarts)
			
			#if ( (l>0) and (math.isclose(L[l],L[l-1],rel_tol=1e-9) ) ):
				#break
		end = timer()

		Optimal_K_em[restarts,K-Min_K] = max_log_likelihood

		iterations_for_each_restart_em[restarts] = total_iterations_em
		print('iterations:', total_iterations_em, 'max_log_likelihood:', "%.15f" % max_log_likelihood, 'time:', end-start)
		
		#vis.clear_plot()
		#vis.visualize_gmm(x1,y1,x2,y2,x3,y3,mu,sigma,5.0,color_data=False, iter='final', restart=restarts, save_plot='False')
		

		
		#vis.clear_plot()
		
		#vis.visualize_log_likelihood(L,restart=restarts, save_plot='True')



		#plt.clf()

		
		
		#initialize responsibility matrix KxN
		R = np.zeros((K,N), np.float64)

		#initialize gaussian means
		mu = np.array(list(initial_mu), np.float64)
		#print(mu.shape)

		#initialize gaussian covariances
		#sigma = np.array([ [ [1,0],[0,1] ],[ [1,0],[0,1] ],[ [1,0],[0,1] ] ], np.float64)
		#sigma = np.array([ [ [5,0],[0,5] ],[ [5,0],[0,5] ],[ [5,0],[0,5] ] ], np.float64)
		sigma = 2 * test_data.k_ten_dimensional_sigma(K)

		
		#initialize loglikelihood array
		#L = np.zeros(Iterations,)
		D = []


		#vis.visualize_kmeans_via_em(x1,y1,x2,y2,x3,y3,mu,sigma,5.0, iter='init',restart=restarts, save_plot='True')
		total_iterations_em = 0
		max_log_likelihood = gmm.compute_log_likelihood(X,mu,sigma,pi,N,K)
		log_likelihood = gmm.compute_log_likelihood(X,mu,sigma,pi,N,K)
		
		total_distance = 0
		#K-means using mahalonobis
		
		start = timer()

		for l in range(0,Iterations):
			plt.clf()
			

			#E-step
			#update responsibility matrix R -> KxN
			R,total_distance = k_means.update_responsibility_matrix_hard_assignment(K,N,X,mu,sigma)
			D.append(total_distance)
			D_kmeans_maha_total[restarts,l] = total_distance
		
			#M-step
			for k in range(0,K):
				mu[k] = k_means.update_mu_k(k,N,R,X,mu[k])
				sigma[k] = k_means.update_sigma_k(R,k,mu,sigma[k],N,X)
			
			
			#log_likelihood = gmm.compute_log_likelihood(X,mu,sigma,pi,N,K)
			log_likelihood = k_means.compute_log_likelihood(X,mu,sigma,N,K)
			L_kmeans_total[restarts,l] = log_likelihood
			if (log_likelihood > max_log_likelihood):
				max_log_likelihood = log_likelihood
			#vis.visualize_kmeans_via_em(x1,y1,x2,y2,x3,y3,mu,sigma,.1,color_data=False, iter=l,restart=restarts)



			total_iterations_em = total_iterations_em + 1

			#if (total_iterations_em > 2 and  D[l] == D[l-1]):
				#break
			
		end = timer()
		
		Optimal_K_kmeans_maha[restarts,K-Min_K] = total_distance
		iterations_for_each_restart_kmeans_maha[restarts] = total_iterations_em
		print('iterations:', total_iterations_em, 'total_mahalanobis_distance_to_cluster_centers:', total_distance, 'time:', end-start)
		#print('iterations:', total_iterations_em, 'max_log_likelihood:', "%.15f" % max_log_likelihood, 'time:', end-start)
		
		
		
		#vis.clear_plot()
		#vis.visualize_kmeans_via_em(x1,y1,x2,y2,x3,y3,mu,sigma,5.0,color_data=False,iter='final', restart=restarts, save_plot='True')
			
	   
		#vis.clear_plot()
		
		#vis.visualize_total_maha_distance(D,restart=restarts, save_plot='True')



		#plt.clf()
		#plt.ion()

		D = []

		#initialize gaussian means
		mu = list(initial_mu)

		#sigma_fixed = [ [ [.05,0],[0,.05] ],[ [.05,0],[0,.05] ],[ [.05,0],[0,.05] ] ]

		#vis.visualize_kmeans(x1,y1,x2,y2,x3,y3,mu,sigma_fixed,5.0, iter='init',restart=restarts, save_plot='True')

		R = np.zeros((K,N))

		

		total_iterations_kmeans = 0
		total_distance = 0

		#K-means algorithm
		start = timer()
		for l in range(0,Iterations):
			#plt.clf()

			
			R,total_distance = k_means.assign_to_clusters(K,N,X,mu)
			D_kmeans_total[restarts,l] = total_distance
			
			D.append(total_distance)


			for k in range(0,K):
				#Update means
				mu[k] = k_means.update_mu_k(k,N,R,X,mu[k])
			
			
			#vis.visualize_kmeans(cluster_zero_x,cluster_zero_y,cluster_one_x,cluster_one_y,cluster_two_x,cluster_two_y,centroids,sigma_fixed,0.1,color_data=True, iter=l, restart=restarts)
			#vis.visualize_kmeans(x1,y1,x2,y2,x3,y3,mu,sigma_fixed,0.1, iter=l,restart=restarts, save_plot='True')

			total_iterations_kmeans = total_iterations_kmeans + 1

			#if (total_iterations_kmeans > 2 and  D[l] == D[l-1]):
				#break

		end = timer()
		#distribution = [len(cluster_zero_x)/N,len(cluster_one_x)/N,len(cluster_two_x)/N ]

		Optimal_K_kmeans_eucl[restarts,K-Min_K] = total_distance
		iterations_for_each_restart_kmeans_eucl[restarts] = total_iterations_kmeans
		print('iterations:', total_iterations_kmeans, 'total_euclidian_distance_to_cluster_centers', total_distance, 'time', end-start)
		
		#vis.clear_plot()
		#vis.visualize_kmeans(x1,y1,x2,y2,x3,y3,mu,sigma_fixed,5.0, iter='init',restart=restarts, save_plot='True')
		
		#vis.clear_plot()
		
		#vis.visualize_total_eucl_distance(D,restart=restarts, save_plot='True')


		print('')





Optimal_K_em_average = 1/Restarts * np.sum(Optimal_K_em,axis=0)
plt.clf()
vis.visualize_optimal_K_em(Optimal_K_em_average, title='Average_Log_Likelihood', save_plot='True', delay=3)

Optimal_K_kmeans_maha_average = 1/Restarts * np.sum(Optimal_K_kmeans_maha,axis=0)
plt.clf()
vis.visualize_optimal_K_kmeans(Optimal_K_kmeans_maha_average, title='Average_Mahalanobis_Distance', measure='Mahalanobis Distance', save_plot='True', delay=3)

Optimal_K_kmeans_eucl_average = 1/Restarts * np.sum(Optimal_K_kmeans_eucl,axis=0)
plt.clf()
vis.visualize_optimal_K_kmeans(Optimal_K_kmeans_eucl_average, title='Average_Euclidean_Distance', measure='Euclidean Distance',save_plot='True', delay=3)









