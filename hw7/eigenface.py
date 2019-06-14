import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.linalg as LA
import random

def PCA(X, n_components):
    """
        principal component analysis
    """
    # standardize data to ~ N(0,1)
    N, dim = X.shape
    # x = (x - x.mean()) / x.std()
    mean = (np.mean(X, axis=0))
    x = X - mean
    
    # compute covariance matrix
    covariance_mat = np.cov(x.T)
    eig_vals, eig_vecs = LA.eig(covariance_mat)
    
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs,key=lambda k: k[0], reverse=True)
    
    w = np.hstack((eig_pairs[0][1].reshape(dim,1), eig_pairs[1][1].reshape(dim,1)))
    for i in range(2,n_components):
        w = np.hstack((w, eig_pairs[i][1].reshape(dim,1)))
    
    return x, w, mean


def reconstruct_face(centered_data, pc, mean, h, w, img_idx):
    weights = np.dot(centered_data, pc.T)
    centered_vector = np.dot(weights[img_idx, :], pc)
    reconstructed_image = ( mean + centered_vector).reshape(h, w)

    return reconstructed_image

def plot_faces(images, n_row, n_col, fig_num):
    f = plt.figure(num=fig_num, figsize=(2.2 * n_col, 2.2 * n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i], cmap=plt.cm.gray)

if __name__ == "__main__":
    directory = 'att_faces/'
    sub_dir = sorted([x[0] for x in os.walk(directory)][1:])
    image_names = []
    for sd in sub_dir:
        for img in os.listdir(sd):
            image_names.append(sd + '/' + img)
    images = np.array([plt.imread(img) for img in image_names], dtype=np.float64)

    # randomly pick 10 images and plot
    img_idx = np.random.randint(images.shape[0], size=10)
    img_samples = np.array([plt.imread(img) for img in (np.array(image_names)[img_idx])], dtype=np.float64)
    n, n_col = 1, 10
    plot_faces(img_samples, n, n_col, 1)

    # get the eigenfaces
    n_samples, h, w = images.shape
    X = images.reshape(n_samples, h * w)

    n_components = 25
    centered_data, pc, mean = PCA(X, n_components)

    eigenfaces = (pc.T).reshape((n_components, h, w))
    plot_faces(eigenfaces.real, 5, 5, 2)

    # reconstruct the sampled faces with the eigenfaces
    recovered_images = [reconstruct_face(centered_data, pc.T.real, mean, h, w, i) for i in img_idx]
    plot_faces(recovered_images, 1, 10, 3)

    plt.show()