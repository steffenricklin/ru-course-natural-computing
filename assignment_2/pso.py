# Natural Computing
# exercise 1 (b)

import math
import numpy as np
import matplotlib.pyplot as plt
import copy

class PSO_base:
    """Parent PSO class. Can be inherited for different PSO use-cases.
    In most cases it should be sufficient to only change the fitness method.

    Parameters
    ----------
    particles: NumpyArray
        Starting array of particles
    velocities: float, tuple, NumpyArray
        Initial velocity of each particle.
    w: float
        Inertia weight behavior
    r: float
        ...
    alpha: float
        ...
    clip_range: tuple
        Min and max values of particle values. Default: (-1, 1)
    """
    def __init__(self, particles, velocities, w, r, alpha, clip_range, minimize=True):
        """Initializes the PSO
        """
        self.particles = particles
        self.velocities = velocities*np.ones(particles.shape)
        self.w = w
        self.r = r
        self.r_ind = np.random.uniform(0,1,1) if r is None else r[0]
        self.r_social = np.random.uniform(0, 1, 1) if r is None else r[1]
        self.alpha = alpha
        self.clip_min = clip_range[0]
        self.clip_max = clip_range[1]
        self.minimize = minimize
        self.fitness = self.compute_fitness(particles)
        self.personal_best = self.particles
        self.personal_best_fitness = self.fitness
        self.global_best = self.particles[np.argmax(self.personal_best_fitness)]
        self.global_best_fitness = np.max(self.personal_best_fitness)

    def fitness(self, particles):
        """Fitness method is left empty, since most child classes will probably have 
        a different fitness function
        """
        raise NotImplementedError("fitness function must be overridden in subclass")

    def update_v(self, particles):
        """Updates the velocities of the current particles using inertia,
        as well as the personal and social influences."""

        inertia = self.w * self.velocities
        #personal_influence = (self.alpha * self.r) *(self.personal_best - particles)
        #social_influence = (self.alpha * self.r) *(self.global_best - particles)
        personal_influence = (self.alpha * self.r_ind) *(self.personal_best - particles)
        social_influence = (self.alpha * self.r_social) *(self.global_best - particles)
        self.r_ind = np.random.uniform(0,1,1) if self.r is None else self.r[0]
        self.r_social = np.random.uniform(0, 1, 1) if self.r is None else self.r[1]
        self.velocities = inertia + personal_influence + social_influence
    
    def update_bests(self, particles, fitness):
        """Updates the personal best positions for each particles, as well as
        the global best positions.
        """
        # update local best
        has_better_fitness = None
        if self.minimize:
            has_better_fitness = fitness < self.personal_best_fitness
        else:
            has_better_fitness = fitness > self.personal_best_fitness
        for i, is_better in enumerate(has_better_fitness):
            if is_better:
                self.personal_best[i] = copy.deepcopy(particles[i])
                self.personal_best_fitness[i] = copy.deepcopy(fitness[i])
                # update global
                if self.minimize and self.personal_best_fitness[i] < self.global_best_fitness:
                    self.global_best = copy.deepcopy(self.personal_best[i])
                    self.global_best_fitness = copy.deepcopy(self.personal_best_fitness[i])
                elif not self.minimize and self.personal_best_fitness[i] > self.global_best_fitness:
                    self.global_best = copy.deepcopy(self.personal_best[i])
                    self.global_best_fitness = copy.deepcopy(self.personal_best_fitness[i])

    def run(self, n_iter=1):
        """Runs the algorithm for n iterations.
        """
        # set starting best position and fitness
        particles = self.particles
        self.update_bests(particles, self.fitness)
        
        # iterate
        fitness_series = np.zeros((n_iter, particles.shape[0]))
        for i in range(n_iter):
            particles = particles + self.velocities
            if not (self.clip_min is None and self.clip_max is None):
                particles = np.clip(particles, self.clip_min, self.clip_max)
            self.particles = particles
            self.fitness = self.compute_fitness(particles)
            self.update_bests(particles, self.fitness)
            fitness_series[i] = self.fitness
            self.update_v(particles)
        return fitness_series
    
    def show_stats(self, string="Final"):
        """Prints social best fitness and position, as well as 
        the fitness and positions of the current particles
        """
        print(f"For w={self.w}:")
        print(f"\tSocial best fitness: {self.global_best_fitness}")
        print(f"\tSocial best position: {self.global_best}")
        for i in range(self.particles.shape[0]):
            print("\tParticle", i+1)
            print(f"\t\t{string} fitness: {self.curr_fitness[i]}")
            print(f"\t\t{string} positions: {self.particles[i]}")

   
    def plot(self, title=None, figname=""):
        title = title if title is not None else "PSO-gbest: classes and centroids"
        if self.n_features == 2:
            for i in range(self.n_clusters):
                l = f"class {i}"
                plt.scatter(self.data_vectors[self.labels==i, 0], self.data_vectors[self.labels==i, 1], label=l, marker="x", alpha=0.5, s=25)
            plt.scatter(self.global_best[:, 0], self.global_best[:, 1], color="black", label="best centroids")
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.show()
        elif self.n_features > 2:  # make scatter-subplots for higher-dimensional data
            labels = np.unique(self.labels)
            fig, axs = plt.subplots(self.n_features, self.n_features, sharex='col', sharey='row', figsize=(8,8), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
            for i in range(self.n_features):
                for j in range(self.n_features):
                    if i!=j:
                        for c in labels:
                            axs[j, i].scatter(self.data_vectors[self.labels==c, i], self.data_vectors[self.labels==c, j], label=c, marker="x", alpha=0.5, s=25)
                        axs[j, i].scatter(self.global_best[:, i], self.global_best[:, j], color="black", label="best centroids")
                        axs[-1, i].set_xlabel(i)
                        axs[j, 0].set_ylabel(j)
            print("global best.shape:", self.global_best.shape)
            print("global_best:", self.global_best)
            lines, labels = fig.axes[-2].get_legend_handles_labels()
            fig.legend(lines, labels, bbox_to_anchor=(0.315, 0.88))
            plt.suptitle(title)
            plt.savefig("results/PSO_"+figname)
            fig.show()


class PSO_1b(PSO_base):
    """
    """
    def __init__(self, particles, velocities, w, r=0.5, alpha=1, clip_range=(-500, 500)):
        super(PSO_1b, self).__init__(particles, velocities, w, r, alpha, clip_range, minimize=False)
    
    def fitness(self, particles):
        """Computes and returns the fitness of the current particles or any given particles
        """
        fitness = -particles * np.sin(np.sqrt(abs(particles)))
        fitness = np.sum(fitness, axis=1)
        return fitness


# class PSO_gbest(PSO_base):
#     """Adaptation of the PSO_1b class for exercise 3

#     Parameters
#     ----------
#     n_particles: int
#         Number of particles used for clustering
#     v_initial: float, tuple, NumpyArray
#         Initial velocity of each particle. 
#     n_clusters: int
#         Number of clusters that shall be found
#     w: float
#         Inertia weight behavior
#     r: tuple of float
#         r[0] for r_individual and r[1] for r_social
#         ...
#     alpha: float
#         ...
#     clip_range: tuple
#         Min and max values of particle values. Default: (-1, 1)
#     """
#     def __init__(self, data, labels, n_particles=10, v_initial=0.0, n_clusters=2, w=0.7298, r=None, alpha=1.4961,clip_range=(-1, 1), n_features = 2):
#         self.data_vectors = data
#         self.labels = labels
#         self.n_clusters = n_clusters
#         self.n_features = n_features
#         clip_range = np.zeros((2, n_features))
#         clip_range[0] = np.min(data, axis=0)
#         clip_range[1] = np.max(data, axis=0)
#         #particles = np.random.uniform(clip_range[0], clip_range[1], (n_particles, n_clusters, n_features))  
#         mean = np.asarray([np.mean(data[..., i]) for i in range(data.shape[-1])])
#         # print("mean.shape:", mean.shape)
#         # print("mean:", mean)
#         particles = np.random.normal(mean, 1.0, (n_particles, n_clusters, n_features))
#         #par_ind = np.random.choice(data.shape[0], n_particles*n_clusters,  replace=False)
#         #particles = np.reshape(data[par_ind,:], (n_particles,n_clusters,n_features))
#         super().__init__(particles, v_initial, w, r, alpha, clip_range)

#     def compute_fitness(self, particles):
#         """
#         for each data vector z_p
#           (i)   find Euclidean distance d(z_p, m_ij) to all 
#                 cluster centroids C_ij
#           (ii)  assign z_p to cluster C_ij such that 
#                 d(z_p, m_ij) is the minimum for C_ij
#           (iii) calculate the fitness using eq.(8)
#         """
#         fitness = np.zeros(particles.shape[0])
#         for i, particle in enumerate(particles):
#             distances_particle = np.zeros((self.data_vectors.shape[0], self.n_clusters))
#             for j, centroid in enumerate(particle):
#                 distances_particle[:,j] = np.linalg.norm(self.data_vectors - np.ones(self.data_vectors.shape)*centroid, axis = 1)
#             min_distances = np.min(distances_particle, axis=1)
#             assigned_clusters =np.argmin(distances_particle, axis=1)
#             present_clusters, counts = np.unique(assigned_clusters, return_counts=True) 
#             # calculate fitness of this particle
#             J_e = 0
#             n_clusters_found = 0
#             for cluster_i in present_clusters:
#                 J_e += np.sum(min_distances[assigned_clusters==cluster_i]) / np.count_nonzero(assigned_clusters==cluster_i)
#                 n_clusters_found += 1
#             fitness[i] = J_e / n_clusters_found
#         return fitness
            
#     def run(self, n_iter=1):
#         """Runs the algorithm for n iterations.
#         """
#         # set starting best position and fitness
#         self.update_bests(self.particles, self.fitness)
        
#         # iterate
#         fitness_series = np.zeros((n_iter, self.particles.shape[0]))
#         for i in range(n_iter):
#             self.particles = self.particles + self.velocities
#             if not (self.clip_min is None and self.clip_max is None):
#                 self.particles = np.clip(self.particles, self.clip_min, self.clip_max)
#             self.fitness = self.compute_fitness(self.particles)
#             self.update_bests(self.particles, self.fitness)
#             fitness_series[i] = self.fitness
#             self.update_v(self.particles)
#         return fitness_series
   
#     def plot(self, title=None, figname=""):
#         title = title if title is not None else "PSO-gbest: classes and centroids"
#         if self.n_features == 2:
#             for i in range(self.n_clusters):
#                 l = f"class {i}"
#                 plt.scatter(self.data_vectors[self.labels==i, 0], self.data_vectors[self.labels==i, 1], label=l, marker="x", alpha=0.5, s=25)
#             plt.scatter(self.global_best[:, 0], self.global_best[:, 1], color="black", label="best centroids")
#             plt.title(title)
#             plt.xlabel("x")
#             plt.ylabel("y")
#             plt.legend()
#             plt.show()
#         elif self.n_features > 2:  # make scatter-subplots for higher-dimensional data
#             labels = np.unique(self.labels)
#             fig, axs = plt.subplots(self.n_features, self.n_features, sharex='col', sharey='row', figsize=(8,8), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
#             for i in range(self.n_features):
#                 for j in range(self.n_features):
#                     if i!=j:
#                         for c in labels:
#                             axs[j, i].scatter(self.data_vectors[self.labels==c, i], self.data_vectors[self.labels==c, j], label=c, marker="x", alpha=0.5, s=25)
#                         axs[j, i].scatter(self.global_best[:, i], self.global_best[:, j], color="black", label="best centroids")
#                         axs[-1, i].set_xlabel(i)
#                         axs[j, 0].set_ylabel(j)
#             print("global best.shape:", self.global_best.shape)
#             print("global_best:", self.global_best)
#             lines, labels = fig.axes[-2].get_legend_handles_labels()
#             fig.legend(lines, labels, bbox_to_anchor=(0.315, 0.88))
#             plt.suptitle(title)
#             plt.savefig("results/PSO_"+figname)
#             fig.show()
