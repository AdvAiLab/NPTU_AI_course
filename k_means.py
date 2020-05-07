import time

import numpy as np
import matplotlib.pyplot as plt
from util_plot import AddPlot

is_3d = True
ax, point_dim = AddPlot(is_3d).returns

n_cluster_points = 100
cluster_shape = (n_cluster_points, point_dim)
means_K = 4
early_stop_distant = 0.01

# Randomly generate clusters using Normal Distribution (randn)
rand_points1 = 0 + 2 * np.random.randn(*cluster_shape)
rand_points2 = 10 + 3 * np.random.randn(*cluster_shape)
if is_3d:
    rand_points3 = [20, 0, 5] + 2 * np.random.randn(*cluster_shape)
    rand_points4 = [30, 20, 10] + 1.5 * np.random.randn(*cluster_shape)
else:
    rand_points3 = [20, 0] + 2 * np.random.randn(*cluster_shape)
    rand_points4 = [30, 20] + 1.5 * np.random.randn(*cluster_shape)
points_num = n_cluster_points * means_K
all_points = np.concatenate((rand_points1, rand_points2, rand_points3, rand_points4))

# We random choice centroids from points.
rand_indexes = np.random.choice(all_points.shape[0], means_K, replace=False)
centroids = all_points[rand_indexes]

distant_arr = np.zeros((points_num, means_K))
iteration = 0
# Loop until converged.
while True:
    points_per_cluster = [[] for _ in range(means_K)]

    # Get minimal distance between each centroid and each point,
    # then put in the points array per cluster.
    for point in all_points:
        distant_per_centroid = []
        # Calculate distance per point with each centroid.
        for cp in centroids:
            distant_per_centroid.append(np.linalg.norm(cp - point))
        cluster_idx = np.argmin(distant_per_centroid)
        points_per_cluster[cluster_idx].append(point)
        ax.scatter(*point, color="C%d" % cluster_idx, s=50, alpha=0.1)

    centroids_distant = 0

    # Calculate the mean of each cluster to got the new centroid of each cluster
    for i, (cluster, old_centroid) in enumerate(zip(points_per_cluster, centroids)):
        ax.scatter(*old_centroid, color="C%d" % i, marker='+', s=200)
        new_centroid = np.mean(cluster, axis=0)
        # Record distance between new and old centroid in oder to determine convergence.
        centroids_distant += np.linalg.norm(new_centroid - old_centroid)
        ax.scatter(*new_centroid, color="C%d" % i, s=200, marker="*")
        # Update centroid
        centroids[i] = new_centroid

    plt.title("iteration %s, Updated distant=%.4f" % (iteration, centroids_distant))
    plt.draw()
    plt.pause(0.001)
    iteration += 1

    # We assume converged when centroid no more updated
    if centroids_distant < early_stop_distant:
        break
    # Only clear figure on non-last figure
    ax.clear()
# Show end plot forever
plt.show()
