import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

UNCLASSIFIED = False
NOISE = None

def plot_classification(data, classifications, n_class, iteration, data_file):
    colors = ['C0', 'C1', 'C2', 'C3']
    clusters = np.array(classifications)
    for i in range(n_class+1):
        target_data = data[clusters == (i+1)]
        unclassified = data[clusters == UNCLASSIFIED]
        noise = data[clusters == NOISE]
        plt.title('iteration ' + str(iteration + 1))
        plt.scatter(target_data[:,0], target_data[:,1], s=7, c=colors[i])
        plt.scatter(unclassified[:,0], unclassified[:,1], s=7, c='C7')
        plt.scatter(noise[:,0], noise[:,1], s=7, c='C8')
    plt.savefig('./dbscan/dbscan_' + data_file + str(iteration + 1) + '.png')
    plt.pause(0.1)

def eps_neighbors(data, pt_id, eps):
    n = data.shape[0]
    neighbors = []
    for i in range(n):
        if ((LA.norm(data[pt_id,:] - data[i,:])) < eps):
            neighbors.append(i)
    return neighbors

def merge_to_cluster(data, eps, min_pts, classifications, pt_id, cluster_id):
    neighbors = eps_neighbors(data, pt_id, eps)
    # print(len(neighbors))
    if len(neighbors) < min_pts:
        classifications[pt_id] = NOISE
        return False
    else:
        classifications[pt_id] = cluster_id
        for i in neighbors:
            classifications[i] = cluster_id
        
        while len(neighbors) > 0:
            current_point = neighbors[0]
            current_neighbors = eps_neighbors(data, current_point, eps)
            if len(current_neighbors) >= min_pts:
                for i in range(len(current_neighbors)):
                    target_point = current_neighbors[i]
                    if classifications[target_point] == UNCLASSIFIED:
                        neighbors.append(target_point)
                        classifications[target_point] = cluster_id
                    if classifications[target_point] == NOISE:
                        classifications[target_point] = cluster_id
            neighbors = neighbors[1:]
        return True

def dbscan(data, eps, min_pts, data_file):
    cluster_id = 1
    # number of data points
    n = data.shape[0]
    classifications = [UNCLASSIFIED] * n
    for pt_id in range(n):
        iteration = pt_id
        if classifications[pt_id] == UNCLASSIFIED:
            if merge_to_cluster(data, eps, min_pts, classifications, pt_id, cluster_id):
                cluster_id = cluster_id + 1
        # plot
        if (iteration <= 100) and ((iteration % 10) == 0):
            plot_classification(data, classifications, 2, iteration, data_file)
    return classifications

if __name__ == '__main__':
    data_file = 'circle'
    data = np.genfromtxt(data_file + '.txt', delimiter=',')
    eps = 0.1
    min_points = 2
    clusters = dbscan(data, eps, min_points, data_file)

    clusters = np.array(clusters)