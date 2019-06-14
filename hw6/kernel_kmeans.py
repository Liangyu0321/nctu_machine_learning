import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

def plot_classification(data, alpha, n_class, iteration, data_file):
    colors = ['C0', 'C1', 'C2', 'C3']
    for i in range(n_class):
        target_data = data[alpha[:,i] == 1]
        plt.title('iteration ' + str(iteration))
        plt.scatter(target_data[:,0], target_data[:,1], s=7, c=colors[i])
    plt.savefig('./kernel_kmeans/kernal_kmeans_' + str(n_class) + '_class_' + data_file + str(iteration) + '.png')
    plt.pause(1)

def kernel_kmeans(data, n_class, gamma, data_file):
    # number of clusters
    k = n_class
    # number of data
    n = data.shape[0]

    # compute Gram matrix
    gram = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # RBF kernel
            gram[i, j] = np.exp(-gamma * (LA.norm(data[i] - data[j]) ** 2))
            gram[j, i] = gram[i, j]
    
    # initialize alpha
    alpha = np.zeros((n, k))
    alpha[:, 0] = np.random.randint(2, size=n)
    alpha[:, 1] = 1 - alpha[:, 0]

    distances = np.zeros((n, k))

    converge = False
    iteration = 0
    while not converge:
        # number of data points for each class
        n_k = sum(alpha)
        iteration += 1
        
        for i in range(k):
            tmp1 = np.ones(n)
            # tmp1 = np.diag(gram)
            tmp2 = (2 / n_k[i]) * np.sum((np.tile(alpha[:,i].T, (n,1)) * gram), axis=1)
            tmp3 = (n_k[i] ** (-2)) *  np.sum((np.array([alpha[:,i].T,]*n).T * np.tile(alpha[:,i], (n,1))) * gram)
            distances[:,i] = tmp1 - tmp2 +  tmp3
            
        old_alpha = alpha
        for i in range(k):
            alpha[:,i] = 1 * (i == np.argmin(distances, axis=1))
        
        # plot
        plot_classification(data, alpha, n_class, iteration, data_file)

        if np.sum(alpha - old_alpha) == 0:
            converge = True
    return alpha

if __name__ == '__main__':
    data_file = 'circle'
    data = np.genfromtxt(data_file + '.txt', delimiter=',')

    k = 2
    gamma = 50
    alpha = kernel_kmeans(data, k, gamma, data_file)