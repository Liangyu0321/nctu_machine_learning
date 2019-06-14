import numpy as np
import matplotlib.pyplot as plt

# read input data points
def read_input_data(file_name):
    f = open(file_name, 'r')
    data = f.readlines()

    x_list = []
    y_list = []
    for n in data:
        x_list.append(float(n.split(',')[0][:-1]))
        y_list.append(float(n.split(',')[1]))
    f.close()

    return x_list, y_list

def print_function(method_name ,n_base , x_solution):
    # output function in string
    func_str = ''
    for i in range(n_base):
        if i < (n_base - 1):
            func_str += (str(round(x_solution[i, 0], 5)) + 'X^' + str(n_base - 1 - i)) + ' + '
        else:
            func_str += str(round(x_solution[i, 0], 5))
    print('{0}:\nFitting Line: {1}'.format(method_name ,func_str))

def matrix_transpose(A):
    transpose = np.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            transpose[j, i] = A[i, j]
    
    return transpose

def design_matrix(n_data_point, n_base, x_list):
    # TODO: use ndarray
    A = np.zeros((n_data_point, n_base))
    for i in range(n_data_point):
        for j in range(n_base):
            A[i, j] = x_list[i] ** (n_base - 1 -j)
    
    return A

def LU_decomposition(A):
    L = np.eye(A.shape[0])
    tmp = A

    for i in range(A.shape[0] - 1):
        for j in range(A.shape[0] - 1 - i):
            L[i+j+1, i] = tmp[i+j+1, i] / tmp[i, i]
            tmp[i+j+1, :] = tmp[i+j+1, :] - L[i+j+1, i] * A[i, :]
    U = tmp
    return L, U

def forward_substitution(L, b):
    y = np.zeros(b.shape)

    for i in range(y.shape[0]):
        y[i, 0] = b [i, 0]
        for j in range(i):
            y[i, 0] = y[i, 0] - L[i, j] * y[j, 0]
    
    return y

def back_substitution(U, y):
    x = np.zeros(y.shape)

    for i in range(x.shape[0]):
        idx = x.shape[0] - 1 - i
        x[idx, 0] = y[idx, 0]
        for j in range(i):
            x[idx, 0] = x[idx, 0] - U[idx, U.shape[1] - 1 - j] * x[x.shape[0] - 1 - j, 0]
        x[idx, 0] = x[idx, 0] / U[idx, idx]

    return x

def getAnswer(A, b):
    L, U = LU_decomposition(A)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)

    return x

def gradient(A, x, b):
    gradient = np.zeros(x.shape)
    tmp = np.dot(matrix_transpose(A), A)
    gradient = 2 * np.dot(tmp, x) - 2 * np.dot(matrix_transpose(A), b)
    return gradient

def hession(A):
    H = 2 * np.dot(matrix_transpose(A), A)
    return H

def rLSE(A, b, l, n_base):
    x = np.zeros((n_base,1))
    # * is element-wise product
    tmp1 = np.dot((matrix_transpose(A)), A) + l * np.eye(n_base)
    tmp2 = np.dot((matrix_transpose(A)), b)
    x = getAnswer(tmp1, tmp2)
    # print('A.T shape: {0}'.format((A.T).shape))
    # print('b shape: {0}'.format(b.shape))
    # print('(A.T) * b shape: {0}'.format(tmp.shape))
    
    return x


def newton(A, b, n_base, epsilon, max_iter):
    # TODO: generate initial guess x0
    x0 = np.zeros((n_base,1))

    # compute inverse of H
    I = np.eye(A.shape[1])
    H = np.dot(matrix_transpose(A), A)
    H_inverse = np.zeros(H.shape)
    for i in range(I.shape[1]):
        tmp = I[:, i][np.newaxis]
        tmp = getAnswer(H, matrix_transpose(tmp))
        for j in range(H_inverse.shape[0]):
            H_inverse[j, i] = tmp[j, 0]

    H_inverse = (1/2) * H_inverse

    x_new = np.zeros((n_base,1))
    for _ in range(max_iter):
        # update gradient
        gdt = gradient(A, x0, b)
        x_new = x0 - np.dot(H_inverse, gdt)
        if (np.absolute(x_new - x0).all()) < epsilon:
            return x_new
        x0 = x_new
    print('Exceed max iterations. No solution found.')
    return x_new

def square_error(x_list, y_list_gt, n_base, x_solution):
    '''square error'''
    x_list_arr = np.array(x_list)
    y_list_predict = np.array([])
    y_predict = 0

    for i in range(n_base):
        y_predict = y_predict + x_solution[i] * (x_list_arr ** (n_base - 1 - i))
    y_list_predict = np.append(y_list_predict, y_predict)

    y_list_gt_arr = np.array(y_list_gt)
    square_error_arr = (y_list_predict - y_list_gt_arr) * (y_list_predict - y_list_gt_arr)

    return np.sum(square_error_arr)



if __name__ == '__main__':
    # user input
    input_file = input('Input file name: ')
    n_base = int(input('Number of polynomial bases: '))
    l = int(input('Lambda for LSE: '))

    x_list, y_list = read_input_data(input_file)

    n_data_point = len(x_list)

    A = design_matrix(n_data_point, n_base, x_list)
    b = np.zeros((n_data_point, 1))
    for i in range(n_data_point):
        b[i, 0] = y_list[i]

    '''rLSE'''
    x_solution_rLSE = rLSE(A, b, l, n_base)
    err_rLSE = square_error(x_list, y_list, n_base, x_solution_rLSE)
    
    '''Newton's method'''
    x_solution_newton = newton(A, b, n_base, 1, 100)
    err_newton = square_error(x_list, y_list, n_base, x_solution_newton)
    # use the final result
    # get the inverse of (AtA)
    # I = np.eye(A.shape[1])
    # H = np.dot(matrix_transpose(A), A)
    # H_inverse = np.zeros(H.shape)
    # for i in range(I.shape[1]):
    #     tmp = I[:, i][np.newaxis]
    #     tmp = getAnswer(H, matrix_transpose(tmp))
    #     for j in range(H_inverse.shape[0]):
    #         H_inverse[j, i] = tmp[j, 0]
    # tmp = np.dot(H_inverse, matrix_transpose(A)) # (AtA)_inverse (At)
    # x_solution_newton_gold = np.dot(tmp, b)

    print_function('\nLSE', n_base, x_solution_rLSE)
    print('Total error: {0}\n'.format(err_rLSE))
    print_function('Newtons', n_base, x_solution_newton)
    print('Total error: {0}\n'.format(err_newton))

    x_p = np.linspace(min(x_list) - 0.5 * abs(max(x_list)), max(x_list) + 0.5 * abs(max(x_list)), 50)
    ground_truth = lambda x: x**3 + x**2 + 3
    prediction_rLSE = 0
    prediction_newton = 0
    # prediction_newton_gold = 0
    # wrong method
    # for i in range(n_base):
    #     prediction_rLSE = prediction_rLSE + x_solution_rLSE[i, 0] * (x_p ** (n_base - 1 - i))
    #     prediction_newton = prediction_newton + x_solution_newton[i, 0] * (x_p ** (n_base - 1 - i))
        # prediction_newton_gold = prediction_newton_gold + x_solution_newton_gold[i, 0] * (x_p ** (n_base - 1 - i))

    for i in range(n_base):
        prediction_rLSE = prediction_rLSE + x_solution_rLSE[i] * (x_p ** (n_base - 1 - i))
        prediction_newton = prediction_newton + x_solution_newton[i] * (x_p ** (n_base - 1 - i))

    # print_function('Newtons gold', n_base, x_solution_newton_gold)

    plt.plot()
    # plt.plot(x_p, ground_truth(x_p), color='olive')
    plt.scatter(x_list, y_list, c='olive')
    plt.plot(x_p, prediction_rLSE)
    plt.plot(x_p, prediction_newton, color='orange')
    # plt.plot(x_p, prediction_newton_gold, color='red')
    plt.show()