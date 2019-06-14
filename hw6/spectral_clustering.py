import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

def plot_classification(data, classifications, n_class, iteration, data_file):
    colors = ['C0', 'C1', 'C2', 'C3']
    for i in range(n_class):
        target_data = data[classifications == i]
        plt.title('iteration ' + str(iteration))
        plt.scatter(target_data[:,0], target_data[:,1], s=7, c=colors[i])
    plt.savefig('./spectral_clustering/spectral_clustering_' + str(n_class) + '_class_' + data_file + str(iteration) + '.png')
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
        plot_classification(original_data, clusters, k, iteration, data_file)
    
    return clusters

def spectral_clustering(data, n_class):
    # number of class
    k = n_class
    # number of data points
    n = data.shape[0]

    # compute similarity
    # similarity matrix
    w = np.zeros((n,n))

    # Gaussian kernel similarity function
    for i in range(n):
        for j in range(n):
            gamma = 50
            w[i, j] = np.exp(-gamma * (LA.norm(data[i] - data[j]) ** 2))
            if i == j:
                w[i, j] = 0
    
    # compute normalized laplacian
    # L = D - W
    # L = D^{-1/2} L D{-1/2}

    D = np.zeros(w.shape)
    tmp = np.sum(w, axis=1)
    D = np.diag(tmp ** (-0.5))
    L_normalized = D.dot(np.diag(tmp) - w).dot(D)

    eig_val, eig_vec = LA.eig(L_normalized)
    dim = len(eig_val)
    dict_eval = dict(zip(eig_val,range(0,dim)))
    k_e_val = np.sort(eig_val)[1:1+k]
    idx = [dict_eval[k] for k in k_e_val]
    e_val = eig_val[idx]
    e_vec = eig_vec[:,idx]

    X = e_vec / np.sqrt(np.sum(e_vec ** 2, axis = 1)).reshape(n,1)

    return X

if __name__ == '__main__':
    data_file = 'moon'
    data = np.genfromtxt(data_file + '.txt', delimiter=',')

    k = 3
    X = spectral_clustering(data, k)
    clusters = k_means(data, X, k, data_file)