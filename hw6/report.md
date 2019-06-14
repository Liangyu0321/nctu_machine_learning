# ML Homework 6 Report
**k-means clustering, kernel k-means, spectral clustering, DBSCAN**

I**mplementation**

----------
1. k-means clustering

First, initialize n cluster centers using mean and standard deviation.

    def k_means(data, n_cluster, data_file):
        k = n_cluster      # number of clusters
        n = data.shape[0]  # number of data points
        
        # generate random centers
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        centers = np.random.randn(k,data.shape[1]) * std + mean

Then, compute euclidean distances from each data point to all cluster centers, and store the results in numpy array  `distances`. Use `distances` to assign each data point to the closest center (cluster). With the new cluster result, update the cluster centers. Do the above two steps until the centers no longer change.

        centers_old = np.zeros(centers.shape)
        centers_new = np.copy(centers)
        # distances[i, k]: distance of data i to cluster k
        distances = np.zeros((n,k))
        # data i belongs to clusters[i]
        clusters = np.zeros(n)
        iteration = 0
        while (LA.norm(centers_new - centers_old) != 0):
            iteration += 1
            # compute distances from each data to every cluster center
            for i in range(k):
                distances[:,i] = LA.norm(data - centers[i], axis=1)
            # assign each data to the closest center (cluster)
            clusters = np.argmin(distances, axis=1)
            centers_old = np.copy(centers_new)
            # update centers
            for i in range(k):
                centers_new[i,:] = np.mean(data[clusters == i], axis=0)
        return clusters


2. kernel k-means

First, compute the gram matrix using RBF kernel. 

![RBF kernel](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559571805949_image.png)


Gram matrix is symmetric.

![Example of gram matrix](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559571914115_Screenshot+from+2019-06-03+22-24-45.png)



    def kernel_kmeans(data, n_class, gamma):
        k = n_class           # number of clusters
        n = data.shape[0]     # number of data
        # compute Gram matrix
        gram = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                # RBF kernel
                gram[i, j] = np.exp(-gamma * (LA.norm(data[i] - data[j]) ** 2))
                gram[j, i] = gram[i, j]

`alpha` is an indicator matrix of cluster assignment. If `alpha[n,k]` equals 1, `data[n]` belongs to cluster k. If `alpha[n,k]` equals 0, `data[n]` dose NOT belong to cluster k.  Randomly initialize `alpha`.

        # initialize alpha
        alpha = np.zeros((n, k))
        alpha[:, 0] = np.random.randint(2, size=n)
        alpha[:, 1] = 1 - alpha[:, 0]

Compute distances from each data point to all cluster centers using the following formula. Assign each data to the closest center and update clusters. If the `alpha` no longer changes, we are done with the clustering.

![distance function of kernel k-means](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559572475761_Screenshot+from+2019-06-03+22-34-19.png)

        distances = np.zeros((n, k))
        converge = False
        iteration = 0
        while not converge:
            n_k = sum(alpha)   # number of data points for each class
            iteration += 1
            for i in range(k):
                tmp1 = np.ones(n)
                tmp2 = (2 / n_k[i]) * np.sum((np.tile(alpha[:,i].T, (n,1)) * gram), axis=1)
                tmp3 = (n_k[i] ** (-2)) *  np.sum((np.array([alpha[:,i].T,]*n).T * np.tile(alpha[:,i], (n,1))) * gram)
                distances[:,i] = tmp1 - tmp2 +  tmp3
                
            old_alpha = alpha
            for i in range(k):
                alpha[:,i] = 1 * (i == np.argmin(distances, axis=1))
            if np.sum(alpha - old_alpha) == 0:
                converge = True
        return alpha


3. spectral clustering

First construct the Gaussian kernel similarity matrix (RBF kernel).

    def spectral_clustering(data, n_class):
        k = n_class             # number of class
        n = data.shape[0]       # number of data points
        # compute similarity
        # similarity matrix
        w = np.zeros((n,n))
        # Gaussian kernel similarity function
        for i in range(n):
            for j in range(n):
                gamma = 50
                w[i, j] = np.exp(-gamma * (LA.norm(data[i] - data[j]) ** 2))
                if i == j:
                    w[i, j] = 0

Then, we compute Graph Laplacian and the normalized one. Use the normalized Graph Laplacian to get the second and third smallest eigen vectors as our new feature values. Finally, do k-means with the new features of eigen vectors as input data.

![example data](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559573237820_image.png)

![Graph Laplacian](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559573184824_image.png)

![example of Graph Laplacian](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559573203262_image.png)

![normalized Graph Laplacian](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559573424569_Screenshot+from+2019-06-03+22-50-11.png)

        # compute normalized laplacian
        # L = D - W
        # L = D^{-1/2} L D{-1/2}
        D = np.zeros(w.shape)
        tmp = np.sum(w, axis=1)
        D = np.diag(tmp ** (-0.5))
        L_normalized = D.dot(np.diag(tmp) - w).dot(D)
    
        eig_val, eig_vec = LA.eig(L_normalized)
        dim = len(eig_val)
        dict_eval = dict(zip(eig_val,range(0,dim)))
        k_e_val = np.sort(eig_val)[1:1+k]
        idx = [dict_eval[k] for k in k_e_val]
        e_val = eig_val[idx]
        e_vec = eig_vec[:,idx]
        X = e_vec / np.sqrt(np.sum(e_vec ** 2, axis = 1)).reshape(n,1)
        return X


4. DBSCAN

According the algorithm:

![DBSCAN algorithm](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559573864175_Screenshot+from+2019-06-03+22-57-27.png)


`classifications`  of all data points are initialized as `UNCLASSIFIED` . Then, we iterate through all data points. If the current data point is `UNCLASSIFIED`, we check whether we should label it as `NOISE` or form a new cluster.

    def dbscan(data, eps, min_pts):
        cluster_id = 1
        # number of data points
        n = data.shape[0]
        classifications = [UNCLASSIFIED] * n
        for pt_id in range(n):
            if classifications[pt_id] == UNCLASSIFIED:
                if merge_to_cluster(data, eps, min_pts, classifications, pt_id, cluster_id):
                    cluster_id = cluster_id + 1
       return classifications

Get the `neighbors` of the current point. If the number of `neighbors` is smaller than a given threshold, i.e. `imin_pts`, we label the current point as `NOISE`. If not, it is as core point, and we have to expand a new cluster.

    def merge_to_cluster(data, eps, min_pts, classifications, pt_id, cluster_id):
        neighbors = eps_neighbors(data, pt_id, eps)
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

**Initialization of k-means with different method**

----------

Use mean and standard deviation to initialize the centers.

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    centers = np.random.randn(k,data.shape[1]) * std + mean

**Results with n clusters**

----------
1. k-means
- 2 clusters
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569004046_k_means_circle1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569004052_k_means_circle2.png)

![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569065381_k_means_moon1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569065393_k_means_moon2.png)

- 3 clusters
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569346725_k_means_3_class_circle1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569346754_k_means_3_class_circle2.png)

![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569346767_k_means_3_class_moon1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569346781_k_means_3_class_moon2.png)

2. kernel k-means
- 2 clusters
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569422017_kernal_kmeans_circle1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569422026_kernal_kmeans_moon1.png)

3. spectral clustering
- 2 clusters
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569526033_spectral_clustering_circle1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569526027_spectral_clustering_circle2.png)

![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569548290_spectral_clustering_moon1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569548277_spectral_clustering_moon2.png)

- 3 clusters
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569578085_spectral_clustering_3_class_circle1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569578067_spectral_clustering_3_class_circle2.png)

![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569618863_spectral_clustering_3_class_moon1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569618847_spectral_clustering_3_class_moon2.png)

- 4 clusters
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569657664_spectral_clustering_4_class_circle1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569657678_spectral_clustering_4_class_circle2.png)

![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569672096_spectral_clustering_4_class_moon1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569760772_spectral_clustering_4_class_moon10.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569672235_spectral_clustering_4_class_moon18.png)



4. DBSCAN

*gray points are unprocessed data*

![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569819129_dbscan_circle1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569819138_dbscan_circle21.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569819151_dbscan_circle41.png)

![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569819165_dbscan_circle61.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569819181_dbscan_circle81.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569819188_dbscan_circle101.png)

![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569835166_dbscan_moon1.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569835177_dbscan_moon21.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569835184_dbscan_moon41.png)

![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569835190_dbscan_moon61.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569835197_dbscan_moon81.png)
![](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559569835206_dbscan_moon101.png)


**Eigenspace of Graph Laplacian in spectral clustering**

----------
![circle data in original space](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559570210036_spectral_clustering_circle2.png)
![moon data in original space](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559570210052_spectral_clustering_moon2.png)

![circle data in Eigenspace](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559570110784_eigen_space_circle.png)
![moon data in Eigenspace](https://paper-attachments.dropbox.com/s_AC7C3A81CA2D24DEFD4A8B091C49937109A26553ABD2DE51D021EB2ACC0B11F4_1559570110797_eigen_space_moon.png)


When the data are projected to the Eigenspace, they become linearly separable. The data within the same cluster have the same coordinates in the Eigenspace. For instance, the outer ring of the circle data are of the same class in the original space. They correspond to the orange ring in the  Eigenspace.

**Reference**

----------
- [Spectral clustering](https://towardsdatascience.com/spectral-clustering-82d3cff3d3b7)

