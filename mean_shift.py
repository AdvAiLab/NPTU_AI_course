import time

import numpy as np
import matplotlib.pyplot as plt

n_cluster_points = 500
point_dim = 2
cluster_shape = (n_cluster_points, point_dim)

# Randomly generate clusters using Normal Distribution (randn)
rand_points = 0 + 2 * np.random.randn(*cluster_shape)

# We only need random point rather than cluster so using uniform distribution.
centroid = 10 * np.random.rand(point_dim) - 5


def neighborhood_points(xs, x_centroid, dist=3):
    eligible_x = []
    for x in xs:
        distance = np.linalg.norm(x-x_centroid)
        if distance <= dist:
            eligible_x.append(x)

    return np.array(eligible_x)


iteration = 0
while True:

    plt.scatter(rand_points[:, 0], rand_points[:, 1], color="blue", s=50, alpha=0.1)

    plt.scatter(*centroid, color="red", marker="+", s=200)
    eligible_points = neighborhood_points(rand_points, centroid)
    plt.scatter(eligible_points[:, 0], eligible_points[:, 1], color="green", s=50, alpha=0.1)
    new_centroid = np.average(eligible_points, axis=0)
    plt.scatter(*new_centroid, color="red", marker="*", s=200)

    # Record distance between new and old centroid in oder to determine convergence.
    centroids_distant = np.linalg.norm(new_centroid - centroid)
    plt.title("iteration %s, Centroids distant=%s" % (iteration, centroids_distant))

    plt.draw()
    plt.pause(0.2)
    iteration += 1

    # We assume converged when centroid no more updated that same as k-means.
    if centroids_distant < 0.001:
        break

    # Only clear figure on non-last figure
    plt.clf()
    centroid = new_centroid
plt.pause(9999)

