import time

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

points_num = 100
points_dim = 2
points_shape = (points_num, points_dim)

rand_points1 = 0 + 2 * np.random.randn(*points_shape)
rand_points2 = 10 + 3 * np.random.randn(*points_shape)
rand_point3 = [20, 0] + 2*np.random.randn(*points_shape)
points_num *= 3
all_points = np.concatenate((rand_points1, rand_points2, rand_point3))

means_K = 3
rand_indexes = np.random.choice(all_points.shape[0], means_K, replace=False)
centroids = all_points[rand_indexes]

cluster_colors = ["red", "blue", "green"]

distant_arr = np.zeros((points_num, means_K))
iteration = 0
while True:
    cluster_points = [[], [], []]
    plt.title("iteration %s" % iteration)
    for i, (cp, color) in enumerate(zip(centroids, cluster_colors)):
        plt.scatter(*cp, color=color, marker='+', s=200)

        for j, ap in enumerate(all_points):
            distant_arr[j, i] = np.linalg.norm(cp-ap)

    for ap, clusters_distant in zip(all_points, distant_arr):
        color_idx = np.argmin(clusters_distant)
        cluster_points[color_idx].append(ap)
        plt.scatter(*ap, color=cluster_colors[color_idx], s=50, alpha=0.1)

    for i, (cluster, c) in enumerate(zip(cluster_points, cluster_colors)):
        new_centroid = np.average(cluster, axis=0)
        plt.scatter(*new_centroid, color=c, s=200, marker="*")
        centroids[i] = new_centroid

    plt.draw()
    plt.pause(0.2)
    plt.clf()
    iteration += 1
