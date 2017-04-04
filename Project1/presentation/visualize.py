import error_ellipse as error
import matplotlib.pyplot as plt

xmin = -10
xmax = 10
ymin = -10
ymax = 10

def clear_plot():
	plt.clf()

def pause(time=5.0):
	plt.pause(time)

def visualize_log_likelihood(L, title='Log_Likelihood_At_Each_Iteration', delay=15.0,restart=-1,save_plot=False):
	title_string = title + "-" + str(restart)
	plt.title(title_string)
	plt.xlabel('Iteration Step')
	plt.ylabel('Log Likelihood')
	plt.plot(L)
	#if save_plot:
		#plt.savefig(title_string + '.png')
	plt.pause(delay)

def visualize_total_maha_distance(D, title='Total_Mahalanobis_Distance_At_Each_Iteration', delay=15.0, restart=-1,save_plot=False):
	title_string = title + "-" + str(restart)
	plt.title(title_string)
	plt.xlabel('Iteration Step')
	plt.ylabel('Mahalanobis Distance')
	plt.plot(D)
	#if save_plot:
		#plt.savefig(title_string + '.png')
	plt.pause(delay)

def visualize_total_eucl_distance(D, title='Total_Euclidian_Distance_At_Each_Iteration', delay=15.0, restart=-1,save_plot=False):
	title_string = title + "-" + str(restart)
	plt.title(title_string)
	plt.xlabel('Iteration Step')
	plt.ylabel('Euclidian Distance')
	plt.plot(D)
	#if save_plot:
		#plt.savefig(title_string + '.png')
	plt.pause(delay)

def visualize_data_with_true_mu_sigma(x1,y1,x2,y2,x3,y3,mu,sigma,delay,color_data=False, title='Training_Data_With_True_Distribution',save_plot=False):
	axes = plt.gca()
	axes.set_xlim([xmin,xmax])
	axes.set_ylim([ymin,ymax])
	plt.title(title)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plot_data(x1,y1,x2,y2,x3,y3,mu,color_data)
	plot_circle(sigma,mu, vol=.75)
	#if save_plot:
		#plt.savefig(title + '.png')
	plt.pause(delay)	

def visualize_kmeans_via_em(x1,y1,x2,y2,x3,y3,mu,sigma,delay,color_data=False, title='K-means_Using_Mahalanobis_Distance',iter=0, restart=0,save_plot=False):
	axes = plt.gca()
	axes.set_xlim([xmin,xmax])
	axes.set_ylim([ymin,ymax])
	title_string = title + "-"+ str(restart) + "-" + str(iter)
	plt.title(title_string)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plot_data(x1,y1,x2,y2,x3,y3,mu,color_data)
	plot_circle(sigma,mu, vol=.75)
	#if save_plot:
		#plt.savefig(title_string + '.png')
	plt.pause(delay)

def visualize_kmeans(x1,y1,x2,y2,x3,y3,centroids,sigma_fixed,delay,color_data=False, title='K-means_Using_Euclidian_Distance',iter=0, restart=0,save_plot=False):
	axes = plt.gca()
	axes.set_xlim([xmin,xmax])
	axes.set_ylim([ymin,ymax])
	title_string = title + "-"+ str(restart) + "-" + str(iter)
	plt.title(title_string)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plot_data(x1,y1,x2,y2,x3,y3,centroids,color_data)
	plot_circle(sigma_fixed,centroids)
	#if save_plot:
		#plt.savefig(title_string + '.png')
	plt.pause(delay)

def visualize_gmm(x1,y1,x2,y2,x3,y3,mu,sigma,delay, color_data=False, title='GMM_Using_EM_Algorithm',iter=0, restart=0,save_plot=False):
	axes = plt.gca()
	axes.set_xlim([xmin,xmax])
	axes.set_ylim([ymin,ymax])
	title_string = title + "-"+ str(restart) + "-" + str(iter)
	plt.title(title_string)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plot_data(x1,y1,x2,y2,x3,y3,mu, color_data)
	plot_error_ellipse(sigma,mu)
	#if save_plot:
		#plt.savefig(title_string + '.png')
	plt.pause(delay)

def plot_circle(sigma_fixed,centroids,vol=.10):
	error.plot_cov_ellipse(sigma_fixed[0],centroids[0], volume=vol, a=.9, ec=[1,0,0])
	error.plot_cov_ellipse(sigma_fixed[1],centroids[1], volume=vol, a=.9, ec=[1,0,0])
	error.plot_cov_ellipse(sigma_fixed[2],centroids[2], volume=vol, a=.9, ec=[1,0,0])


def plot_data(x1,y1,x2,y2,x3,y3,mu, color_data=False):
	if color_data:
		plt.plot(x1, y1, 'o', color='blue', alpha=0.5)
		plt.plot(x2, y2, 'o', color='green', alpha=0.5)
		plt.plot(x3, y3, 'o', color='black', alpha=0.5)
	else:
		plt.plot(x1, y1, 'o', color='grey', alpha=0.5)
		plt.plot(x2, y2, 'o', color='grey', alpha=0.5)
		plt.plot(x3, y3, 'o', color='grey', alpha=0.5)
	plt.plot(mu[0][0],mu[0][1], 'o', color='red')
	plt.plot(mu[1][0],mu[1][1], 'o', color='red')
	plt.plot(mu[2][0],mu[2][1], 'o', color='red')


def plot_error_ellipse(sigma,mu):
	
	#error.plot_cov_ellipse(sigma[0],mu[0])
	#error.plot_cov_ellipse(sigma[1],mu[1])
	#error.plot_cov_ellipse(sigma[2],mu[2])

	#error.plot_cov_ellipse(sigma[0],mu[0], volume=.25)
	#error.plot_cov_ellipse(sigma[1],mu[1], volume=.25)
	#error.plot_cov_ellipse(sigma[2],mu[2], volume=.25)

	#error.plot_cov_ellipse(sigma[0],mu[0], volume=.75, fc='red')
	#error.plot_cov_ellipse(sigma[1],mu[1], volume=.75, fc='yellow')
	#error.plot_cov_ellipse(sigma[2],mu[2], volume=.75, fc='blue')

	error.plot_cov_ellipse(sigma[0],mu[0], volume=.75, a=.9, ec=[1,0,0])
	error.plot_cov_ellipse(sigma[1],mu[1], volume=.75, a=.9, ec=[1,0,0])
	error.plot_cov_ellipse(sigma[2],mu[2], volume=.75, a=.9, ec=[1,0,0])
