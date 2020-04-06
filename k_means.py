import time

import numpy as np
import matplotlib.pyplot as plt

n_cluster_points = 100
cluster_dim = 2
cluster_shape = (n_cluster_points, cluster_dim)
means_K = 4

# Randomly generate clusters using Normal Distribution (randn)
rand_points1 = 0 + 2 * np.random.randn(*cluster_shape)
rand_points2 = 10 + 3 * np.random.randn(*cluster_shape)
rand_points3 = [20, 0] + 2 * np.random.randn(*cluster_shape)
rand_points4 = [30, 20] + 1.5 * np.random.randn(*cluster_shape)
points_num = n_cluster_points * means_K
all_points = np.concatenate((rand_points1, rand_points2, rand_points3, rand_points4))

# We random choice centroids from points.
rand_indexes = np.random.choice(all_points.shape[0], means_K, replace=False)
centroids = all_points[rand_indexes]

cluster_colors = ["red", "blue", "green", "purple"]

distant_arr = np.zeros((points_num, means_K))
iteration = 0
# Loop until converged.
while True:
    points_per_cluster = [[] for _ in cluster_colors]
    # Calculate distance per point with each centroid.
    for i, (cp, color) in enumerate(zip(centroids, cluster_colors)):
        plt.scatter(*cp, color=color, marker='+', s=200)

        for j, point in enumerate(all_points):
            distant_arr[j, i] = np.linalg.norm(cp - point)

    # Get minimal distance between each centroid and each point, then put in the plotting array per color.
    for point, clusters_distant in zip(all_points, distant_arr):
        color_idx = np.argmin(clusters_distant)
        points_per_cluster[color_idx].append(point)
        plt.scatter(*point, color=cluster_colors[color_idx], s=50, alpha=0.1)

    centroids_distant = 0
    new_centroids = []
    # Calculate the mean of each cluster to got the new centroid of each cluster
    for cluster, color, old_centroid in zip(points_per_cluster, cluster_colors, centroids):
        new_centroid = np.average(cluster, axis=0)
        # Record distance between new and old centroid in oder to determine convergence.
        centroids_distant += np.linalg.norm(new_centroid - old_centroid)
        plt.scatter(*new_centroid, color=color, s=200, marker="*")
        new_centroids.append(new_centroid)
    centroids = new_centroids

    plt.title("iteration %s, Centroids distant=%.4f" % (iteration, centroids_distant))
    plt.draw()
    plt.pause(0.001)
    iteration += 1

    # We assume converged when centroid no more updated
    if centroids_distant < 0.01:
        break
    # Only clear figure on non-last figure
    plt.clf()
plt.pause(9999)
