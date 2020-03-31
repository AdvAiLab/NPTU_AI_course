import time

import numpy as np
import matplotlib.pyplot as plt

cluster_num = 500
cluster_dim = 2
cluster_shape = (cluster_num, cluster_dim)

rand_points = 0 + 2 * np.random.randn(*cluster_shape)
centroid = 10 * np.random.rand(cluster_dim) - 5


def neighbourhood_points(xs, x_centroid, dist=3):
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
    eligible_points = neighbourhood_points(rand_points, centroid)
    plt.scatter(eligible_points[:, 0], eligible_points[:, 1], color="green", s=50, alpha=0.1)
    new_centroid = np.average(eligible_points, axis=0)
    plt.scatter(*new_centroid, color="red", marker="*", s=200)

    centroids_distant = np.linalg.norm(new_centroid - centroid)
    plt.title("iteration %s, Centroids distant=%s" % (iteration, centroids_distant))

    plt.draw()
    plt.pause(0.2)
    plt.clf()
    iteration += 1

    if centroids_distant < 0.001:
        break
    centroid = new_centroid
plt.pause(9999)

