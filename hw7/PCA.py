import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def PCA(X=None, n_pc=2):
    """
        principal component analysis
    """
    N, dim = X.shape
    # center each column (feature) of data to zero mean
    x = X - (np.mean(X, axis=0))
    # compute covariance matrix and eigenvalues, eigenvectors
    covariance_mat = np.cov(x.T)
    eig_vals, eig_vecs = LA.eig(covariance_mat)
    # rank the eigenvectors (principal components by eigenvalues)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    # in descending order
    eig_pairs = sorted(eig_pairs,key=lambda k: k[0], reverse=True)

    # expleined variance
    print('variance explained:\n')
    eigv_sum = sum(eig_vals)
    for i, j in enumerate(eig_pairs):
        print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
    
    w = np.hstack((eig_pairs[0][1].reshape(dim,1), eig_pairs[1][1].reshape(dim,1)))
    for i in range(2,n_pc):
        w = np.hstack((w, eig_pairs[i][1].reshape(dim,1)))
    
    return x.dot(w)

if __name__ == "__main__":
    # read data
    data_file = 'mnist_X.csv'
    label_file = 'mnist_label.csv'
    X = np.genfromtxt(data_file, delimiter=',')
    y = np.genfromtxt(label_file, delimiter=',')

    labels, counts = (np.unique(y, return_counts=True))
    labels = labels.astype(int)

    n_pc = 3
    x_PCA = PCA(X=X, n_pc=n_pc)

    # plot
    if n_pc == 2:
        for l,c in zip(labels,('C0', 'C1', 'C2', 'C3', 'C4')):
            target_data = x_PCA[y == l]
            plt.scatter(target_data[:,0].real, target_data[:,1].real, c=c, s=7, alpha=0.5)
        plt.legend(labels.astype(int), loc='upper left')
    elif n_pc == 3:
        fig = plt.figure()
        ax = Axes3D(fig)

        for l,c in zip(labels,('C0', 'C1', 'C2', 'C3', 'C4')):
            target_data = x_PCA[y == l]
            plt.scatter(target_data[:,0].real, target_data[:,1].real, c=c, s=7, alpha=0.5)
        plt.legend(labels.astype(int), loc='upper left')
    else:
        print('The dimension is larger than 3.')

    plt.show()