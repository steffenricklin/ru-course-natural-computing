"""Implementation of the k-means clustering for exercise 3
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class KMeansClustering():
    """Implementation of the k-means clustering for exercise 3
    """
    def __init__(self,nr_input_dimensions, nr_data_points,nr_clusters, data, labels):
        self.Nd = nr_input_dimensions
        self.No = nr_data_points
        self.Nc = nr_clusters
        self.vectors = data
        self.labels = labels
        self.centroids = np.zeros((self.Nc,self.Nd))

    def calculate_fitness(self, assigned_clusters, centroids, min_dist):
        J_e = 0
        for cluster_i in range(self.Nc):
            J_e += np.sum(min_dist[assigned_clusters==cluster_i]) / np.count_nonzero(assigned_clusters==cluster_i)
        fitness = J_e / self.Nc
        return fitness   

    def k_means(self, nr_iter):
        #randomly initialize Nc centroids in the data
        assigned_cluster = np.zeros(self.No).astype(int)
        centroids = np.zeros((self.Nc,self.Nd))
        centroids_indx = np.random.choice(self.No, self.Nc, replace=False)
        for i in range(centroids.shape[0]):
            a =centroids_indx[i]
            centroids[i] = self.vectors[a]       
        fitness = np.zeros(nr_iter)

        
        for i in range(nr_iter):
            #compute euclidean distance between data points and centroids
            # and assign closest cetroids accordingly
            min_dist = np.zeros(self.No)
            for p,z_p in enumerate(self.vectors):
                min_centr_nr = assigned_cluster[p]
                min_dist[p] = np.linalg.norm(z_p-centroids[min_centr_nr,:]) 
                for centr_nr, centroid in enumerate(centroids):
                    dist = np.linalg.norm(z_p-centroid)
                    if dist<min_dist[p]:
                        min_dist[p] = dist
                        assigned_cluster[p] = centr_nr
                        
            #calculate fitness
            fitness[i] = self.calculate_fitness(assigned_cluster, centroids, min_dist)

            #recalculate the cetroids by computing the
            for centr_nr, centroid in enumerate(centroids):            
                centroids[centr_nr] = np.average(self.vectors[assigned_cluster == centr_nr], axis = 0)
        self.centroids = centroids
        return assigned_cluster, centroids, fitness
    

    def plot(self, title=None, figname=""):
        title = title if title is not None else "K-means: classes and centroids"
        if self.Nd == 2:
            for i in range(self.Nc):
                l = f"class {i}"
                plt.scatter(self.vectors[self.labels==i, 0], self.vectors[self.labels==i, 1], label=l, marker="x", alpha=0.5, s=25)
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color="black", label="best centroids")
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.show()
        elif self.Nd > 2:  # make scatter-subplots for higher-dimensional data
            labels = np.unique(self.labels)
            fig, axs = plt.subplots(self.Nd, self.Nd, sharex='col', sharey='row', figsize=(8,8), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
            for i in range(self.Nd):
                for j in range(self.Nd):
                    if i!=j:
                        for c in labels:
                            axs[j, i].scatter(self.vectors[self.labels==c, i], self.vectors[self.labels==c, j], label=c, marker="x", alpha=0.5, s=25)
                        axs[j, i].scatter(self.centroids[:, i], self.centroids[:, j], color="black", label="best centroids")
                        axs[-1, i].set_xlabel(i)
                        axs[j, 0].set_ylabel(j)
            print("global best.shape:", self.centroids.shape)
            print("global_best:", self.centroids)
            lines, labels = fig.axes[-2].get_legend_handles_labels()
            fig.legend(lines, labels, bbox_to_anchor=(0.315, 0.88))
            plt.suptitle(title)
            plt.savefig("results/Kmeans_"+figname)
            fig.show()
    

