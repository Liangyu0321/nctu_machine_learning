import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

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

def polynomial_basis_linear_model_gen(n_basis, a, w, x_range, plus_err):
    if plus_err:
        err = univariate_gaussian_data_gen(0, a)
    else:
        err = 0
    x = np.random.uniform(-x_range, x_range)
    # print('x: %f' % x)
    poly = np.poly1d(w[::-1])
    # print('poly: {}'.format(poly))
    y = poly(x) + err

    return x, y

def update_mean_variance(x_new, mean_old, variance_old, count):
    mean_new = mean_old + (x_new - mean_old) / count
    # variance_new = variance_old + ((x_new - mean_old) ** 2) / count - variance_old / (count - 1)
    variance_new = variance_old + ((x_new - mean_old) * (x_new - mean_new) - variance_old) / count

    return mean_new, variance_new

def online_algorithm(m_src, s_src, mean, variance, count):
    x = univariate_gaussian_data_gen(m_src, s_src)
    count = count + 1
    m_new, s_new = update_mean_variance(x, mean, variance, count)

    print('Add data point: {0}'.format(x))
    print('Mean = %f Variance = %f\n' % (m_new, s_new))

    if abs(m_new - mean) > 0.005 or abs(s_new - variance) > 0.005:
        online_algorithm(m_src, s_src, m_new, s_new, count)

def baysian_linear_regression(n, a, w):
    count = 0
    data_points = []
    posterior_means, posterior_variances = [], []
    while(1):
        # generate one data point
        x, y = polynomial_basis_linear_model_gen(n, a, w, 1, True)
        data_points.append([x, y])
        count += 1
        print('Add data point #%d (%f, %f):\n' % (count, x, y))
        
        # calculate posterior mean and variance
        x_vector = np.zeros(n)
        for i in range(n):
            x_vector[i] = x ** i
        x_vector = x_vector.reshape((1, -1))

        if count == 1:
            var_inv = a * np.dot(x_vector.T, x_vector) + b * np.eye(n)
            mean = a * np.dot(inv(var_inv), x_vector.T) * y
        else:
            var_inv = a * np.dot(x_vector.T, x_vector) + p_s
            mean = np.dot(inv(var_inv), (a * x_vector.T * y + np.dot(p_s, p_m)))
        
        if count == 10 or count == 50:
            posterior_means.append(mean)
            posterior_variances.append(inv(var_inv))
        
        print('Posterior mean:\n{}\n'.format(mean))
        print('Posterior variance:\n{}\n'.format(inv(var_inv)))
        
        # inverse of prior's covariance matrix
        p_s = var_inv
        # prior's mean
        p_m = mean

        # predictive distribution
        predictive_m = np.dot(x_vector, mean)
        predictive_var = 1 / a + np.dot(np.dot(x_vector, inv(var_inv)), x_vector.T)
        print('Predictive distribution ~ N(%f, %f)' % (predictive_m, predictive_var))
        print('-' * 30)
        if count > 1:
            if abs(previous_predictive_var - predictive_var) < 0.000001:
                posterior_means.append(mean)
                posterior_variances.append(inv(var_inv))
                break
        previous_predictive_var = predictive_var
    
    return data_points, posterior_means, posterior_variances

if __name__ == '__main__':

    problem_number = int(input('Problem number: '))
    if problem_number == 1:
        # problem 1.a
        # univariate gaussian data generator
        m = float(input('Expectation value or mean: '))
        s = float(input('Variance: '))

        x = univariate_gaussian_data_gen(m, s)
        print('A data point from N({0}, {1}): {2}\n'.format(m, s, x))

        # problem 1.b
        # polynomial basis linear model data generator
        n = int(input('Basis number (n): '))
        a = float(input('Variance for error term: '))
        w = input('Coefficients of the polynomial (n x 1 vector): ')
        
        # convert w from string to list of int elements
        w = list(map(int, w[1:-1].split(',')))
        
        _, y = polynomial_basis_linear_model_gen(n, a, w, 1, True)
        print('output: {}'.format(y))
    
    elif problem_number == 2:
        # sequential estimator (for-loop version)
        m_src = float(input('Expectation value or mean: '))
        s_src = float(input('Variance: '))
        print('Data point source function: N({}, {})\n'.format(m_src, s_src))
        
        # ramdonly assign initial mean and variance
        m_initial, s_initial = 1, 5
        online_algorithm(m_src, s_src, m_initial, s_initial, 0)
    
    elif problem_number == 3:
        # problem 3
        # Baysian linear regression
        b = int(input('Precision (b): '))
        n = int(input('Basis number (n): '))
        a = float(input('Variance for error term (a): '))
        w = input('Coefficients of the polynomial (w, n x 1 vector): ')
        
        # convert w from string to list of int elements
        w = list(map(int, w[1:-1].split(',')))

        data_points, posterior_means, posterior_variances = baysian_linear_regression(n, a, w)
        # plot the result
        # 1. ground truth
        num_points = 1000
        ground_truth = np.zeros((num_points, 2))
        for i in range(num_points):
            ground_truth[i, 0], ground_truth[i, 1] = polynomial_basis_linear_model_gen(n, a, w, 2, False)
        
        # sort the array according to x value
        ground_truth_sort = ground_truth[ground_truth[:,0].argsort()]

        num_x = 500
        xs = np.linspace(-2, 2, num=num_x)
        xs_vector = np.zeros((num_x, n))
        for i in range(num_x):
            for j in range(n):
                xs_vector[i, j] = xs[i] ** j
        
        # after 10 incomes
        means_vector_10 = np.zeros(num_x)
        variance_vector_10 = np.zeros(num_x)
        for i in range(num_x):
            means_vector_10[i] = np.dot(xs_vector[i], posterior_means[0])
            variance_vector_10[i] = np.dot(np.dot(xs_vector[i], posterior_variances[0]), xs_vector[i].reshape((1,-1)).T)
        
        # after 50 incomes
        means_vector_50 = np.zeros(num_x)
        variance_vector_50 = np.zeros(num_x)
        for i in range(num_x):
            means_vector_50[i] = np.dot(xs_vector[i], posterior_means[1])
            variance_vector_50[i] = np.dot(np.dot(xs_vector[i], posterior_variances[1]), xs_vector[i].reshape((1,-1)).T)

        # final predictive result
        means_vector_final = np.zeros(num_x)
        variance_vector_final = np.zeros(num_x)
        for i in range(num_x):
            means_vector_final[i] = np.dot(xs_vector[i], posterior_means[2])
            variance_vector_final[i] = np.dot(np.dot(xs_vector[i], posterior_variances[2]), xs_vector[i].reshape((1,-1)).T)
        
        data_points = np.array(data_points)
        
        f, ax = plt.subplots(2, 2)
        
        ax[0, 0].set_title('Ground truth')
        ax[0, 0].plot(ground_truth_sort[:, 0], ground_truth_sort[:, 1], 'k')
        ax[0, 0].plot(ground_truth_sort[:, 0], ground_truth_sort[:, 1] + 1/a, 'r')
        ax[0, 0].plot(ground_truth_sort[:, 0], ground_truth_sort[:, 1] - 1/a, 'r')
        
        ax[1, 0].set_title('After 10 incomes')
        ax[1, 0].plot(data_points[:10, 0], data_points[:10, 1], 'bo')
        ax[1, 0].plot(xs, means_vector_10, 'k')
        ax[1, 0].plot(xs, means_vector_10 + variance_vector_10, 'r')
        ax[1, 0].plot(xs, means_vector_10 - variance_vector_10, 'r')
        
        ax[1, 1].set_title('After 50 incomes')
        ax[1, 1].plot(data_points[:50, 0], data_points[:50, 1], 'bo')
        ax[1, 1].plot(xs, means_vector_50, 'k')
        ax[1, 1].plot(xs, means_vector_50 + variance_vector_50, 'r')
        ax[1, 1].plot(xs, means_vector_50 - variance_vector_50, 'r')
        
        ax[0, 1].set_title('Predict result')
        ax[0, 1].plot(data_points[:, 0], data_points[:, 1], 'bo')
        ax[0, 1].plot(xs, means_vector_final, 'k')
        ax[0, 1].plot(xs, means_vector_final + variance_vector_final, 'r')
        ax[0, 1].plot(xs, means_vector_final - variance_vector_final, 'r')
        
        plt.ylim(-20, 20)
        plt.show()
