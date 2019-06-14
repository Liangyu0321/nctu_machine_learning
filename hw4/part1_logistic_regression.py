import numpy as np
import matplotlib.pyplot as plt
from random import randint
from numpy.linalg import inv, det

def univariate_gaussian_data_gen(m, s):
    # univariate gaussian data generator
    # Box–Muller method
    U, V = np.random.uniform(0, 1, 2)
    # z ~ N(0, 1) standard normal distribution
    z = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V)
    # z = (x - mean)/standard_deviaition
    standard_deviaition = np.sqrt(s)
    x = standard_deviaition * z + m

    return x

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def gradient_ascent(X, Y, D1_D2):

    w = np.array([randint(0,5), randint(0,5), randint(0,5)]).reshape(1,-1).T
    for _ in range(10000):
        tmp = Y - sigmoid(np.dot(X, w))
        cost = np.dot(X.T, tmp)
        w_n = w + cost
        
        # print(w_n - w)
        if abs(w_n - w).all() < 1:
            w = w_n
            break
        
        w = w_n
    
    prediction = sigmoid(np.dot(X, w)).astype(int)
    # print(prediction)
    # print((prediction.flatten()).shape)
    idx = np.argwhere(prediction == 0)[:,0]
    predict_D1 = D1_D2[idx]
    idx = np.argwhere(prediction == 1)[:,0]
    predict_D2 = D1_D2[idx]

    print('Gradient descent:\n')
    print('w:\n{}\n'.format(w))
    calc_confusion_matrix(prediction.flatten(), (Y.astype(int)).flatten())

    return (predict_D1, predict_D2)

def newtons_method(X, Y, D1_D2, N):
    # Newton's method
    w = np.array([randint(0,5), randint(0,5), randint(0,5)]).reshape(1,-1).T
    learning_rate = 0.1
    
    for _ in range(10000):
        # compute Hession function of J
        numerator = np.exp(-np.dot(X, w))
        denominator = sigmoid(np.dot(X, w)) * sigmoid(np.dot(X, w))
        
        D = np.zeros((N * 2, N * 2))
        for i in range(N * 2):
            D[i, i] = numerator[i] / denominator[i]

        H = np.dot(np.dot(X.T, D), X)

        tmp = Y - sigmoid(np.dot(X, w))
        cost = np.dot(X.T, tmp)

        if (det(H) != 0):
            # print('use Newtons')
            w_n = w + learning_rate * np.dot(inv(H), cost)
        else:
            w_n = w + cost
        
        if abs(w_n - w).all() < 1:
            w = w_n
            break
        
        w = w_n
    
    p = sigmoid(np.dot(X, w)).astype(int)
    idx = np.argwhere(p == 0)[:,0]
    predict_D1 = D1_D2[idx]
    idx = np.argwhere(p == 1)[:,0]
    predict_D2 = D1_D2[idx]
    
    print('Newtons method:\n')
    print('w:\n{}\n'.format(w))
    calc_confusion_matrix(p.flatten(), (Y.astype(int)).flatten())

    return (predict_D1, predict_D2)

def calc_confusion_matrix(prediction, ground_truth):
    confusion_mat = np.zeros((2, 2))
    for p in range(2):
        for gt in range(2):
            confusion_mat[p, gt] = np.sum(ground_truth[np.argwhere(prediction == p)] == gt)
    confusion_mat = confusion_mat.astype(int)
    sensitivity = confusion_mat[0,0] / (confusion_mat[0,0] + confusion_mat[1,0])
    specificity = confusion_mat[0,1] / (confusion_mat[0,1] + confusion_mat[1,1])
    print('Confusion matrix:')
    print('{:20}{:^20}{:^20}'.format(' ','Is cluster 0', 'Is cluster 1'))
    print('{:<20}{:^20d}{:^20d}'.format('Predict cluster 0', confusion_mat[0, 0], confusion_mat[0, 1]))
    print('{:<20}{:^20d}{:^20d}\n'.format('Predict cluster 1', confusion_mat[1, 0], confusion_mat[1, 1]))
    print('{}: {}'.format('Sensitivity: ', sensitivity))
    print('{}: {}'.format('Specificity: ', specificity))

if __name__ == '__main__':

    mx1, my1 = 1, 1
    vx1, vy1 = 2, 2
    mx2, my2 = 10, 10
    vx2, vy2 = 2, 2
    N = 50 # number of data points

    D1, D2 = [], []
    for _ in range(N):
        x1 = univariate_gaussian_data_gen(mx1, vx1)
        y1 = univariate_gaussian_data_gen(my1, vy1)
        x2 = univariate_gaussian_data_gen(mx2, vx2)
        y2 = univariate_gaussian_data_gen(my2, vy2)
        # last element as the ground truth of the data
        D1.append([x1, y1, 0])
        D2.append([x2, y2, 1])

    D1 = np.array(D1)
    D2 = np.array(D2)

    Y = np.concatenate((D1[:,2], D2[:,2])).reshape(-1, 1)
    D1 = D1[:, 0:2]
    D2 = D2[:, 0:2]

    x0 = np.ones((N * 2, 1))
    D1_D2 = np.vstack((D1, D2))
    X = np.concatenate((x0, D1_D2), axis=1)

    gradient_ascent_prediction = gradient_ascent(X, Y, D1_D2)
    print('-' * 30)
    newtons_method_prediction = newtons_method(X, Y, D1_D2, N)

    # plot
    _, ax = plt.subplots(1, 3)
    ax[0].set_title('Ground truth')
    ax[0].scatter(D1[:, 0], D1[:, 1])
    ax[0].scatter(D2[:, 0], D2[:, 1], marker='x')

    ax[1].set_title('Gradient descent')
    ax[1].scatter(gradient_ascent_prediction[0][:, 0], gradient_ascent_prediction[0][:, 1])
    ax[1].scatter(gradient_ascent_prediction[1][:, 0], gradient_ascent_prediction[1][:, 1], marker='x')
    
    ax[2].set_title('Newtons method')
    ax[2].scatter(newtons_method_prediction[0][:, 0], newtons_method_prediction[0][:, 1])
    ax[2].scatter(newtons_method_prediction[1][:, 0], newtons_method_prediction[1][:, 1], marker='x')
    plt.show()