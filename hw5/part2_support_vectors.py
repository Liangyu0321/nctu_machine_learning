from svmutil import *
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def rbfKernel(x, y):
    gamma = 3e-2
    ans = np.exp(-gamma * (cdist(x, y, metric='euclidean') ** 2))
    return ans

if __name__ == '__main__':
    # read data as LIBSVM format (dict)
    y_train_dict, x_train_dict = svm_read_problem('part2_libsvm_data.csv')
    prob = svm_problem(y_train_dict, x_train_dict)
    # read data as numpy array
    x_train_np = np.genfromtxt('Plot_X.csv', delimiter=',')
    y_train_np = np.genfromtxt('Plot_Y.csv', delimiter=',')

    # linear kernel
    param = svm_parameter('-s 0 -t 0')
    model_linear = svm_train(prob, param)
    # get indices of support vectors
    sv_idx_linear = np.array(model_linear.get_sv_indices())

    # polynomial kernel
    param = svm_parameter('-s 0 -t 1 -d 2')
    model_polynomial = svm_train(prob, param)
    sv_idx_polynomial = np.array(model_polynomial.get_sv_indices())

    # rbf kernel
    param = svm_parameter('-s 0 -t 2')
    model_rbf = svm_train(prob, param)
    sv_idx_rbf = np.array(model_rbf.get_sv_indices())

    # linear + RBF
    tmp = 0.1 * np.dot(x_train_np, x_train_np.T) + 100 * rbfKernel(x_train_np, x_train_np)
    K = np.c_[ np.arange(1, x_train_np.shape[0]+1).reshape(-1,1), tmp]
    model_linear_rbf = svm_train(y_train_np, K, '-s 0 -t 4')
    sv_idx_linear_rbf = np.array(model_linear_rbf.get_sv_indices())

    # plot
    sv_indices = [sv_idx_linear, sv_idx_polynomial, sv_idx_rbf, sv_idx_linear_rbf]
    labels = np.unique(y_train_np).astype(int)
    colors = ['C1', 'C3', 'C0']
    markers = ['1', 'x', '*']
    titles = ['Linear', 'Polynomial', 'RBF', 'Linear + RBF']
    _, ax = plt.subplots(2, 2)
    for i, sv_idx in enumerate(sv_indices):
        for idx, l in enumerate(labels):
            target_data = x_train_np[y_train_np == l]
            ax[int(i/2), i-2].set_title(titles[i] + ' kernel')
            ax[int(i/2) , i-2].scatter(target_data[:,0], target_data[:,1], c=colors[l], s=1)
            # plot support vectors
            tmp = sv_idx[sv_idx > idx*1000]
            sv = tmp[tmp < (idx+1)*1000]
            for idx in sv:
                ax[int(i/2), i-2].scatter(x_train_np[idx-1,0], x_train_np[idx-1,1], c=colors[l], marker=markers[l])
    plt.show()
                    