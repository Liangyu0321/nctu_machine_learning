import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt

def load_data(X_filename, label_filename):
    print('load data.....')
    x = np.genfromtxt(X_filename, delimiter=',')
    y = np.genfromtxt(label_filename, delimiter=',')
    return x, y

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    # Implement PCA here
    # standardize data to ~ N(0,1)
    X = (X - X.mean()) / X.std()

    # compute covariance matrix
    covariance_mat = np.cov(X.T)
    eig_vals, eig_vecs = LA.eig(covariance_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs,key=lambda k: k[0], reverse=True)

    dim = X.shape[1]
    w = np.hstack((eig_pairs[0][1].reshape(dim,1), eig_pairs[1][1].reshape(dim,1)))
    for i in range(2,no_dims):
        w = np.hstack((w, eig_pairs[i][1].reshape(dim,1)))

    X_PCA = X.dot(w)

    return X_PCA

def neg_squared_euclidean_dists(x):
    """
        Compute negative squared eclidean distances for all pairs of points
        in matrix x.
    """
    sum_x = np.sum(np.square(x), 1)
    dists = np.add(np.add(-2. * np.dot(x, x.T), sum_x).T, sum_x)

    return -dists

def q_joint(Y):
    """
        Given low-dimensional representations Y, compute matrix of joint 
        probabilities with entries q_ij.
    """
    # compute distances from every y_i to y_j
    dists = neg_squared_euclidean_dists(Y)
    exp_dists = np.exp(dists)
    # let q_ii = 0
    np.fill_diagonal(exp_dists, 0.)
    # divide by the sum of the exponential matrix
    Q = exp_dists / np.sum(exp_dists)

    return Q, None

def grad_symmetric_sne(P, Q, Y, _):
    pq_diff = P - Q                                         # N x N
    pq_expand = np.expand_dims(pq_diff, 2)                  # N x N x 1
    y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)    # N x N x 2
    grad = 2. * (pq_expand * y_diff).sum(1)                 # N x 2

    return grad

def q_tsne(Y):
    """
        Given low-dimensional representations Y, compute matrix of joint 
        probabilities with entries q_ij in student t-distribution.
    """
    dists = -neg_squared_euclidean_dists(Y)
    inv_dists = (1. + dists) ** -1
    np.fill_diagonal(inv_dists, 0.)
    Q = inv_dists / np.sum(inv_dists)

    return Q, inv_dists

def grad_tsne(P, Q, Y, dists):
    pq_diff = P - Q                                         # N x N
    pq_expand = np.expand_dims(pq_diff, 2)                  # N x N x 1
    y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)    # N x N x 2
    dists_expand = np.expand_dims(dists, 2)                 # N x N x 1
    y_diff_weighted = y_diff * dists_expand                 # N x N x 2
    grad = 4. * (pq_expand * y_diff_weighted).sum(1)        # N x 2

    return grad

def sne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, q_func=q_joint, grad_func=grad_symmetric_sne):
    """
        Runs symmetric SNE or t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        Q, dists = q_func(Y)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        dY = grad_func(P, Q, Y, dists)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y

if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X, y = load_data('mnist_X.csv', 'mnist_label.csv')
    labels, counts = (np.unique(y, return_counts=True))
    labels = labels.astype(int)
    
    TSNE = False
    Y_symmetric_sne = sne(X, 2, 50, 20.0, q_func=q_tsne if TSNE else q_joint, grad_func=grad_tsne if TSNE else grad_symmetric_sne)
    
    TSNE = True
    Y_tsne = sne(X, 2, 50, 20.0, q_func=q_tsne if TSNE else q_joint, grad_func=grad_tsne if TSNE else grad_symmetric_sne)

    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.title('Symmetric SNE')
    for l,c in zip(labels,('C0', 'C1', 'C2', 'C3', 'C4')):
        target_data = Y_symmetric_sne[y == l]
        plt.scatter(target_data[:, 0], target_data[:, 1], c=c, s=7, alpha=0.5)
    plt.legend(labels.astype(int), loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.title('t-SNE')
    for l,c in zip(labels,('C0', 'C1', 'C2', 'C3', 'C4')):
        target_data = Y_tsne[y == l]
        plt.scatter(target_data[:, 0], target_data[:, 1], c=c, s=7, alpha=0.5)
    plt.legend(labels.astype(int), loc='upper left')
    
    plt.show()