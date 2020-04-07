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

centroid_arr = np.copy(all_points)

while True:
    plt.scatter(all_points[:, 0], all_points[:, 1], color="blue", s=50, alpha=0.1)
    plt.scatter(centroid_arr[:, 0], centroid_arr[:, 1], color="red", s=50, alpha=0.1)

    distant_list = []
    for i, rp in enumerate(all_points):
        centroid = centroid_arr[i]

        eligible_points = neighborhood_points(all_points, centroid, dist=5)
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
    if mean_distance < 0.1:
        break
    iteration += 1

centroid_arr = np.sort(np.mean(centroid_arr, axis=1))
centroid_diff = np.diff(centroid_arr)
cluster_idx = np.argwhere(centroid_diff > 5)
splited_cluster = np.split(all_points, cluster_idx.astype(int).ravel())
splited_centroid = np.split(centroid_arr, cluster_idx)

for cluster, centroids, color in zip(splited_cluster, splited_centroid, cluster_colors):
    new_centroid = np.mean(centroids)
    plt.scatter(*new_centroid, color=color, marker="*", s=200, alpha=1.0)
    plt.scatter(cluster[:, 0], cluster[:, 1], color=color, s=50, alpha=0.1)
plt.title("Clustering result")
plt.show()
plt.pause(9999)

