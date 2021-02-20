"""Contains used data sets
 - Artificial 1 from '"Data clustering using particle swarm optimization."
                      Evolutionary Computation, 2003.'

"""
import numpy as np
import matplotlib.pyplot as plt


class artificial_problem_1():
    """Implementing and testing in the notebook.
    Final version will be moved here from notebook once finished."""
    def __init__(self, n_vectors, sample_range=(-1,1)):
        low, high = sample_range
        self.data = np.random.uniform(low, high, (n_vectors, 2))
        z1_ge07 = self.data[:, 0] >= 0.7
        z1_le03 = self.data[:, 0] <= 0.3
        z2_ge = self.data[:, 1] >= -0.2-self.data[:, 0]
        log_and = np.logical_and(z1_le03, z2_ge)
        self.is_class_1 = np.logical_or(z1_ge07, log_and)
        self.labels = np.zeros((n_vectors), dtype=int)
        self.labels[self.is_class_1] = 1

    def get_samples(self):
        return self.data, self.labels

    def plot(self):
        plt.scatter(self.data[self.is_class_1, 0], self.data[self.is_class_1, 1], label="class1", marker="x", alpha=0.5, s=25)
        plt.scatter(self.data[~self.is_class_1, 0], self.data[~self.is_class_1, 1], label="class0", marker="x", alpha=0.5, s=25)
        plt.title("Artificial Problem 1")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.legend()
        plt.show()

