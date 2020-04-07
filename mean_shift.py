import time
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

n_cluster_points = 100
point_dim = 2
cluster_shape = (n_cluster_points, point_dim)

# Randomly generate clusters using Normal Distribution (randn)
rand_points = 0 + 2 * np.random.randn(*cluster_shape)
rand_points2 = 10 + 3 * np.random.randn(*cluster_shape)
rand_points3 = [20, 0] + 2 * np.random.randn(*cluster_shape)
rand_points4 = [20, 20] + 1.5 * np.random.randn(*cluster_shape)

all_points = np.concatenate((rand_points, rand_points2, rand_points3, rand_points4), axis=0)

# We only need random point rather than cluster so using uniform distribution.
# centroid = 10 * np.random.rand(point_dim) - 5
cluster_colors = ["red", "blue", "green", "purple"]


def neighborhood_points(xs, x_centroid, dist=3):
    eligible_x = []
    for x in xs:
        distance = np.linalg.norm(x-x_centroid)
        if distance <= dist:
            eligible_x.append(x)

    return np.array(eligible_x)


iteration = 0

# Make all points as centroids
centroid_arr = np.copy(all_points)

while True:
    plt.scatter(all_points[:, 0], all_points[:, 1], color="blue", s=50, alpha=0.1)
    plt.scatter(centroid_arr[:, 0], centroid_arr[:, 1], color="red", s=50, alpha=0.1)

    distant_list = []
    for i, rp in enumerate(all_points):
        centroid = centroid_arr[i]

        eligible_points = neighborhood_points(all_points, centroid, dist=6)
        new_centroid = np.mean(eligible_points, axis=0)

        # Record distance between new and old centroid in oder to determine convergence.
        distant_list.append(np.linalg.norm(new_centroid - centroid))

        # Only clear figure on non-last figure
        centroid_arr[i] = new_centroid

    mean_distance = mean(distant_list)
    plt.title("iteration %s, mean_distance=%.4f" % (iteration, mean_distance))
    plt.pause(0.5)
    plt.draw()
    plt.clf()
    # We assume converged when centroid no more updated that same as k-means.
    if mean_distance < 0.0001:
        break
    iteration += 1


def direct_calculate():
    sorted_all = all_points[ind]
    splited_cluster = np.split(sorted_all, cluster_idx.ravel())

    for i, (cluster, centroids) in enumerate(zip(splited_cluster, splited_centroid)):
        new_centroid = np.mean(centroids, axis=0)
        plt.scatter(*new_centroid, color="C%d" % i, marker="*", s=200, alpha=1.0)
        plt.scatter(cluster[:, 0], cluster[:, 1], color="C%d" % i, s=50, alpha=0.1)


def k_means(centroids, points_num):
    distant_arr = np.zeros(points_num, len(centroids))
    # Calculate distance per point with each centroid.
    for i, (cp, color) in enumerate(centroids):
        plt.scatter(*cp, color="C%d" % i, marker='+', s=200)

        for j, point in enumerate(all_points):
            distant_arr[j, i] = np.linalg.norm(cp - point)

    # Get minimal distance between each centroid and each point.
    for point, clusters_distant in zip(all_points, distant_arr):
        color_idx = np.argmin(clusters_distant)
        plt.scatter(*point, color="C%d" % color_idx, s=50, alpha=0.1)


ind = np.lexsort((centroid_arr[:, 1], centroid_arr[:, 0]))
sorted_centroids = centroid_arr[ind]
centroid_diff = np.linalg.norm(np.diff(sorted_centroids, axis=0), axis=1)
cluster_idx = np.argwhere(centroid_diff > 1)
splited_centroid = np.split(sorted_centroids, cluster_idx.ravel())


plt.title("Clustering result: %s cluster" % (i+1))
plt.draw()
plt.pause(9999)

