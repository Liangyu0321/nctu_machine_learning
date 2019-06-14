from svmutil import *
import numpy as np
from scipy.spatial.distance import cdist

def rbfKernel(x, y):
    gamma = 3e-2
    ans = np.exp(-gamma * (cdist(x, y, metric='euclidean') ** 2))
    return ans

if __name__ == '__main__':
    # read data as dictionary
    y_train_dict, x_train_dict = svm_read_problem('part1_libsvm_train_data.csv')
    prob = svm_problem(y_train_dict, x_train_dict)
    y_test_dict, x_test_dict = svm_read_problem('part1_libsvm_test_data.csv')
    # read data as numpy array
    x_train_np = np.genfromtxt('X_train.csv', delimiter=',')
    y_train_np = np.genfromtxt('Y_train.csv', delimiter=',')
    x_test_np = np.genfromtxt('X_test.csv', delimiter=',')
    y_test_np = np.genfromtxt('Y_test.csv', delimiter=',')

    # C-SVC, linear kernel
    param = svm_parameter('-s 0 -t 0')
    model_linear = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_test_dict, x_test_dict, model_linear)

    # C-SVC, polynomial kernel
    param = svm_parameter('-s 0 -t 1 -d 2')
    model_polynomial = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_test_dict, x_test_dict, model_polynomial)

    # C-SVC, rbf kernel
    param = svm_parameter('-s 0 -t 2')
    model_rbf = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_test_dict, x_test_dict, model_rbf)

    # linear + RBF kernel
    tmp = 0.1 * np.dot(x_train_np, x_train_np.T) + 100 * rbfKernel(x_train_np, x_train_np)
    K_train = np.c_[ np.arange(1, x_train_np.shape[0]+1).reshape(-1,1), tmp]
    tmp = 0.1 * np.dot(x_test_np, x_train_np.T) + 100 * rbfKernel(x_test_np, x_train_np)
    K_test = np.c_[ np.arange(1, x_test_np.shape[0]+1).reshape(-1,1), tmp]

    model = svm_train(y_train_np, K_train, '-s 0 -t 4')
    p_label, p_acc, p_val = svm_predict(y_test_np, K_test, model)