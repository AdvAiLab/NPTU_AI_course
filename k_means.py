import time

import numpy as np
import matplotlib.pyplot as plt

n_cluster_points = 100
cluster_dim = 2
cluster_shape = (n_cluster_points, cluster_dim)
means_K = 4


def spawn_points():
    rand_points1 = 0 + 2 * np.random.randn(*cluster_shape)
    rand_points2 = 10 + 3 * np.random.randn(*cluster_shape)
    rand_points3 = [20, 0] + 2 * np.random.randn(*cluster_shape)
    rand_points4 = [30, 20] + 1.5 * np.random.randn(*cluster_shape)
    points_num = n_cluster_points * means_K
    all_points = np.concatenate((rand_points1, rand_points2, rand_points3, rand_points4))
    return points_num, all_points


points_num, all_points = spawn_points()

rand_indexes = np.random.choice(all_points.shape[0], means_K, replace=False)
centroids = all_points[rand_indexes]

cluster_colors = ["red", "blue", "green", "purple"]

distant_arr = np.zeros((points_num, means_K))
iteration = 0
while True:
    cluster_points = [[] for _ in cluster_colors]
    for i, (cp, color) in enumerate(zip(centroids, cluster_colors)):
        plt.scatter(*cp, color=color, marker='+', s=200)

        for j, point in enumerate(all_points):
            distant_arr[j, i] = np.linalg.norm(cp - point)

    for point, clusters_distant in zip(all_points, distant_arr):
        color_idx = np.argmin(clusters_distant)
        cluster_points[color_idx].append(point)
        plt.scatter(*point, color=cluster_colors[color_idx], s=50, alpha=0.1)

    centroids_distant = 0
    for i, (cluster, c) in enumerate(zip(cluster_points, cluster_colors)):
        new_centroid = np.average(cluster, axis=0)
        centroids_distant += np.linalg.norm(new_centroid - centroids[i])
        plt.scatter(*new_centroid, color=c, s=200, marker="*")
        centroids[i] = new_centroid
    plt.title("iteration %s, Centroids distant=%s" % (iteration, centroids_distant))

    plt.draw()
    plt.pause(0.2)
    plt.clf()
    iteration += 1

    if centroids_distant < 0.01:
        break
