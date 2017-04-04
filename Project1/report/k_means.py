import numpy as np

import random
import math
import sys

def assign_to_clusters(K,N,X,mu):
	
	R = np.zeros((K,N))
	total_distance_to_cluster_center = 0
	for i in range(0,N):
		arg_min = 0
		min_distance = sys.maxsize
		distance = 0
		for k in range(0,K):
			
			#distance = compute_mahalanobis_distance(X[i],centroids[k],sigma_fixed[k])
			distance = compute_euclidian_distance(X[i],mu[k])
			if(distance < min_distance):
				arg_min = k
				min_distance = distance
		R[arg_min,i] = 1
		total_distance_to_cluster_center = total_distance_to_cluster_center + min_distance
		#this is just to visualize
		#assign_to_cluster_lists(arg_min,i)
			
	return R,total_distance_to_cluster_center


def update_mu_k(k,N,R,X,mu_k):
	D = len(mu_k)
	sum_rik_xi = np.zeros(D)

	#sum_rik_xi = [0,0]
	rk = 0
		
	for i in range(0,N):
		sum_rik_xi = sum_rik_xi + R[k,i]*X[i]
		rk = rk + R[k,i]
	if( rk == 0):
		#if cluster k has no membership then don't try and move it
		return mu_k
	else:
		return sum_rik_xi/rk

	return sum_rik_xi/rk

def update_sigma_k(R,k,mu,sigma_k,N,X):
	D = len(sigma_k)
	small_diagonal = 1e-1 * np.eye(D,dtype=float)
	sum_rik_xi_xi_T = small_diagonal
	rk = 0
	for i in range(0,N):
		sum_rik_xi_xi_T =  sum_rik_xi_xi_T + R[k,i]* np.outer ((X[i] - mu[k]),(X[i] - mu[k]))
		rk = rk + R[k,i]

	if rk == 0:
		return sigma_k

	sigmak = sum_rik_xi_xi_T/rk 
	
	return sigmak 	

def update_responsibility_matrix_hard_assignment(K,N,X,mu,sigma):
	
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
			
	return R,total_distance_to_cluster_center


def compute_mahalanobis_distance(xi,muk,sigmak):
	sigmak_inv = np.linalg.inv(sigmak)
	d = np.sqrt(np.dot(np.dot(np.transpose((xi-muk) ),sigmak_inv), (xi-muk) ))
	return d

def compute_euclidian_distance(xi,muk):
	d = [(a - b)**2 for a, b in zip(xi, muk)]
	d = math.sqrt(sum(d))
	return d

def probability(xi,muk,sigmak):
	D = len(muk)
	
	sigma_inverse = np.linalg.inv(sigmak)
	
	distance = (-1/2) * ((xi-muk).T.dot(sigma_inverse)).dot((xi-muk))
	det_sigmak = np.linalg.det(sigmak)
	normalizing_constant = 1 / ( ((2* np.pi)**(D/2)) * (det_sigmak**(1/2))  )
	weight = float(normalizing_constant * np.exp(distance))

	return weight

def compute_log_likelihood(X,mu,sigma,N,K):
	sum = 0
	for i in range(0,N):
		inner_sum = 0
		for k in range(0,K):
			inner_sum = inner_sum + probability(X[i],mu[k],sigma[k])
		sum = sum + np.log(inner_sum)
	return sum
