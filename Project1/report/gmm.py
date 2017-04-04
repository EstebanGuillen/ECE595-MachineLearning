import numpy as np



def weighted_probability(xi,muk,sigmak,pik):
	D = len(muk)
	
	sigma_inverse = np.linalg.inv(sigmak)
	
	distance = (-1/2) * ((xi-muk).T.dot(sigma_inverse)).dot((xi-muk))
	det_sigmak = np.linalg.det(sigmak)
	normalizing_constant = 1 / ( ((2* np.pi)**(D/2)) * (det_sigmak**(1/2))  )
	weight = pik * float(normalizing_constant * np.exp(distance))

	return weight


def update_responsibility_matrix(R,X,mu,sigma,pi,N,K):
	for i in range(0,N):
		sum = 0
		prob = 0
		for k in range(0,K):
			prob = weighted_probability(X[i],mu[k],sigma[k],pi[k])
			R[k,i] = prob
			sum = sum + prob
	
		#normalize over the clusters, each cluster should sum to 1 
		for k in range(0,K):
			#make sure to not divide by zero, shouldn't happen but just being safe
			if(sum != 0):
				R[k,i] = R[k,i]/sum
			else:
				R[k,i] = 0
	return R

def update_mu_k(R,k,mu_k,N,X):
	D = len(mu_k)
	sum_rik_xi = np.zeros(D)
	#sum_rik_xi = [0,0]
	rk = 0
		
	for i in range(0,N):
		sum_rik_xi = sum_rik_xi + R[k,i]*X[i]
		rk = rk + R[k,i]
	if rk == 0:
		#if cluster k has no membership then don't try and move it
		return mu_k
	else:
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

def update_pi_k(R,k,N):
	rk = 0
	for i in range(0,N):
		rk = rk + R[k,i]
	return rk/N

def compute_log_likelihood(X,mu,sigma,pi,N,K):
	sum = 0
	for i in range(0,N):
		inner_sum = 0
		for k in range(0,K):
			inner_sum = inner_sum + weighted_probability(X[i],mu[k],sigma[k],pi[k])
		sum = sum + np.log(inner_sum)
	return sum

