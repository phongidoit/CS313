import numpy as np
from scipy.spatial.distance import pdist, squareform

# Helper function to compute the distance matrix
def DistanceMatrix(data, metric):
    '''
    This function computes the pairwise Euclidean distance matrix for the data.
    
    Arguments
    ---------
    data - numpy array or pandas DataFrame of shape (n_samples, n_features)
    
    Returns
    -------
    Distance matrix of shape (n_samples, n_samples)
    '''
    if metric=="manhattan":
        metric="cityblock"
    return squareform(pdist(data, metric=metric))

# DianaClustering Class
class DianaClustering:
    def __init__(self, data, metric='euclidean'): 
        '''
        Constructor of the class, it takes the main data frame as input
        '''
        self.data = data  
        self.n_samples, self.n_features = data.shape
        self.children_ = []  # To save the history of splits
        self.metric = metric

    def fit(self, n_clusters):
        '''
        This method uses the main Divisive Analysis algorithm to do the clustering

        Arguments
        ---------
        n_clusters - integer
                     Number of clusters we want

        Returns
        -------
        cluster_labels - numpy array
                         An array where the cluster number of a sample corresponding 
                         to the same index is stored
        '''
        similarity_matrix = DistanceMatrix(self.data, self.metric)  # similarity matrix of the data
        clusters = [list(range(self.n_samples))]       # list of clusters, initially the whole dataset is a single cluster
        while True:
            # Compute cluster diameters and find the cluster with max diameter
            c_diameters = [np.max(similarity_matrix[cluster][:, cluster]) for cluster in clusters] 
            max_cluster_dia = np.argmax(c_diameters)
            
            # Find the index with the max mean difference within the cluster
            max_difference_index = np.argmax(np.mean(similarity_matrix[clusters[max_cluster_dia]][:, clusters[max_cluster_dia]], axis=1))
            splinters = [clusters[max_cluster_dia][max_difference_index]]  # Splinter group
            last_clusters = clusters[max_cluster_dia]
            del last_clusters[max_difference_index]
            
            while True:
                split = False
                for j in range(len(last_clusters))[::-1]:
                    splinter_distances = similarity_matrix[last_clusters[j], splinters]
                    last_distances = similarity_matrix[last_clusters[j], np.delete(last_clusters, j, axis=0)]
                    if np.mean(splinter_distances) <= np.mean(last_distances):
                        splinters.append(last_clusters[j])
                        del last_clusters[j]
                        split = True
                        break
                if not split:
                    break
            
            # Save the current split to children_
            self.children_.append((splinters, last_clusters))
            
            # Update clusters
            del clusters[max_cluster_dia]
            clusters.append(splinters)
            clusters.append(last_clusters)
            
            if len(clusters) == n_clusters:
                break

        # Create the final labels
        cluster_labels = np.zeros(self.n_samples)
        for i, cluster in enumerate(clusters):
            cluster_labels[cluster] = i

        return cluster_labels