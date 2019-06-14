import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

def plot_classification(data, classifications, centers_new, n_class, iteration, data_file):
    colors = ['C0', 'C1', 'C2', 'C3']
    for i in range(n_class):
        target_data = data[classifications == i]
        plt.title('iteration ' + str(iteration))
        plt.scatter(target_data[:,0], target_data[:,1], s=7, c=colors[i])
        plt.scatter(centers_new[i,0], centers_new[i,1], s=100 ,marker='X', c=colors[i], edgecolors='C3')
    plt.savefig('./k_means/k_means_' + str(n_class) + '_class_' + data_file + str(iteration) + '.png')
    plt.pause(1)

def k_means(original_data, data, n_cluster, data_file):
    # number of clusters
    k = n_cluster
    # number of data
    n = data.shape[0]
    
    # generate random centers
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    centers = np.random.randn(k,data.shape[1]) * std + mean
    
    centers_old = np.zeros(centers.shape)
    centers_new = np.copy(centers)

    # distances[i, k]: distance of data i to cluster k
    distances = np.zeros((n,k))
    # data i belongs to clusters[i]
    clusters = np.zeros(n)
    
    iteration = 0
    while(LA.norm(centers_new - centers_old) != 0):
        iteration += 1
        # compute distances from each data to every cluster center
        for i in range(k):
            distances[:,i] = LA.norm(data - centers[i], axis=1)
        # assign each data to the closest center (cluster)
        clusters = np.argmin(distances, axis=1)

        centers_old = np.copy(centers_new)
        # update centers
        for i in range(k):
            centers_new[i,:] = np.mean(data[clusters == i], axis=0)
        
        # plot current classificaton
        plot_classification(original_data, clusters, centers_new, k, iteration, data_file)
    
    return clusters

if __name__ == '__main__':
    data_file = 'circle'
    data = np.genfromtxt(data_file + '.txt', delimiter=',')

    k = 3
    clusters = k_means(data, data, k, data_file)