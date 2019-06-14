import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def LDA(x=None, n_ld=2):
    """
        linear discriminant analysis
    """
    N, dim = x.shape
    # 1. compute the d-dimentsional mean vectors
    mean_vectors = []
    for l in labels:
        # column mean, i.e. mean of each feature
        mean_vectors.append(np.mean(x[y == l], axis=0))
    
    # 2. compute scatter matrices
    # within-class scatter matrix: s_w
    s_w = np.zeros((dim, dim))
    for l, mv in zip(labels, mean_vectors):
        scatter_mat = np.zeros((dim, dim))
        for d in x[y == l]:
            d = d.reshape(dim, 1)
            mv = mv.reshape(dim, 1)
            scatter_mat += (d - mv).dot((d - mv).T)
        s_w += scatter_mat
    
    # between-class scatter matrix: s_b
    # mean of each feature
    overall_mean = np.mean(x, axis=0)
    s_b = np.zeros((dim, dim))
    for l, mv in zip(labels, mean_vectors):
        # n is the number of data with label l
        n = x[y == l].shape[0]
        overall_mean = overall_mean.reshape(dim, 1)
        mv = mv.reshape(dim, 1)
        s_b += n * (mv - overall_mean).dot((mv - overall_mean).T)
    
    # 3. sovle inv(s_w)s_b
    # check if s_w is a singularmatrix
    if LA.det(s_w) == 0:
        # use pseudo inverse
        eig_vals, eig_vecs = LA.eig(LA.pinv(s_w).dot(s_b))
    else:
        eig_vals, eig_vecs = LA.eig(LA.inv(s_w).dot(s_b))

    # 4. select linear discriminants
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs,key=lambda k: k[0], reverse=True)

    w = np.hstack((eig_pairs[0][1].reshape(dim,1), eig_pairs[1][1].reshape(dim,1)))
    for i in range(2,n_ld):
        w = np.hstack((w, eig_pairs[i][1].reshape(dim,1)))
    
    return x.dot(w)

if __name__ == "__main__":
    # read data
    data_file = 'mnist_X.csv'
    label_file = 'mnist_label.csv'
    x = np.genfromtxt(data_file, delimiter=',')
    y = np.genfromtxt(label_file, delimiter=',')

    labels, counts = (np.unique(y, return_counts=True))
    labels = labels.astype(int)

    n_ld = 2
    x_LDA = LDA(x=x, n_ld=n_ld)

    # plot
    if n_ld == 2:
        for l,c in zip(labels,('C0', 'C1', 'C2', 'C3', 'C4')):
            target_data = x_LDA[y == l]
            plt.scatter(target_data[:,0].real, target_data[:,1].real, c=c, s=7, alpha=0.5)
        plt.legend(labels.astype(int), loc='upper left')
    elif n_ld == 3:
        fig = plt.figure()
        ax = Axes3D(fig)

        for l,c in zip(labels,('C0', 'C1', 'C2', 'C3', 'C4')):
            target_data = x_LDA[y == l]
            plt.scatter(target_data[:,0].real, target_data[:,1].real, c=c, s=7, alpha=0.5)
        plt.legend(labels.astype(int), loc='upper left')
    else:
        print('The dimension is larger than 3.')

    plt.show()