import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import os

class kmeans:
    '''Implementing Kmeans algorithm'''

    def __init__(self, drivers, max_iter=100, random_state=123):
        self.n_clusters = len(drivers)
        self.max_iter = max_iter
        self.random_state = random_state
        self.drivers = drivers

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, pickups):
        X = np.array(pickups)
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)

    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)


class modified_kmeans:
    '''Implementing modified kmeans algorithm'''

    def __init__(self, drivers_home, drivers_carload, max_iter=100, random_state=123):
        self.n_clusters = len(drivers_home)
        self.max_iter = max_iter
        self.random_state = random_state
        self.drivers_home = drivers_home
        self.drivers_carload = drivers_carload
        self.n_points_per_cluster = []

    def initializ_centroids(self):
        centroids = self.drivers_home
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            points_to_consider = np.vstack([X[labels == k, :], self.drivers_home[k, :]])
            centroids[k, :] = np.mean(points_to_consider, axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        dist_dict = self.create_dict_from_distance(distance)
        sorted_dist = {k: v for k, v in sorted(dist_dict.items(), key=lambda item: item[1])}
        num_points = distance.shape[0]
        labels = self.match_label_by_capacity(sorted_dist, num_points)
        return labels

    def match_label_by_capacity(self, sorted_dist, num_points):
        updated_carload = self.drivers_carload.copy()
        labeled_points = []
        labels = -1 * np.ones(num_points)
        for key in sorted_dist:
            if len(labeled_points) == num_points:
                break
            i_point, j_cluster = key[0], key[1]
            if i_point not in labeled_points and updated_carload[j_cluster] > 0:
                labels[i_point] = j_cluster
                labeled_points.append(i_point)
                updated_carload[j_cluster] -= 1
        if any(labels < 0):
            print('Error: some points did not get label')
        return labels


    def create_dict_from_distance(self, distance):
        dist_dict = {}
        for i_point in range(distance.shape[0]):
            for j_centroid in range(distance.shape[1]):
                    dist_dict[(i_point, j_centroid)] = distance[i_point, j_centroid]
        return dist_dict

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, pickups):
        X = np.array(pickups)
        self.centroids = self.initializ_centroids()
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
        self.n_points_per_cluster = [np.count_nonzero(self.labels == k) for k in range(self.n_clusters)]

    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)


def plot_clusters(points_data, cluster_label, drivers_home_and_cluster, x_label='', y_label='', figure_name=''):
    """x-y graph"""
    n_clusters = len(drivers_home_and_cluster)
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))
    c_map = plt.cm.get_cmap('jet', n_clusters)
    plt.scatter(drivers_home_and_cluster[:, 0], drivers_home_and_cluster[:, 1], s=150, cmap=c_map, c=np.array(range(n_clusters)))
    plt.scatter(points_data[:, 0], points_data[:, 1], s=15, cmap=c_map, c=cluster_label)
    plt.colorbar()
    plt.xlabel(x_label), plt.ylabel(y_label)
    plt.title(figure_name)

    # my_path = os.path.abspath(r'C:\Users\user\Desktop\University\Unsupervised_learning\figures')  # Figures out the absolute path for you in case your working directory moves around.
    # my_file = method_name
    # plt.savefig(os.path.join(my_path, my_file))