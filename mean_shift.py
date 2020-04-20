import numpy as np
import matplotlib.pyplot as plt
from util_3d import add_plot

is_3d = True
ax, point_dim = add_plot(is_3d)

n_cluster_points = 100
cluster_shape = (n_cluster_points, point_dim)

# Randomly generate clusters using Normal Distribution (randn)
rand_points1 = 0 + 2 * np.random.randn(*cluster_shape)
rand_points2 = 10 + 3 * np.random.randn(*cluster_shape)
if is_3d:
    rand_points3 = [20, 0, 5] + 2 * np.random.randn(*cluster_shape)
    rand_points4 = [20, 20, 10] + 1.5 * np.random.randn(*cluster_shape)
else:
    rand_points3 = [20, 0] + 2 * np.random.randn(*cluster_shape)
    rand_points4 = [20, 20] + 1.5 * np.random.randn(*cluster_shape)

all_points = (rand_points1, rand_points2, rand_points3, rand_points4)
points_num = n_cluster_points * len(all_points)
all_points = np.concatenate(all_points, axis=0)


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

# Iterate until converged
while True:
    ax.scatter(*all_points.T, color="blue", s=50, alpha=0.1)
    ax.scatter(*centroid_arr.T, color="red", s=50, alpha=0.1)

    distant_list = []
    for i, rp in enumerate(all_points):
        centroid = centroid_arr[i]

        eligible_points = neighborhood_points(all_points, centroid, dist=6)
        new_centroid = np.mean(eligible_points, axis=0)

        # Record distance between new and old centroid in oder to determine convergence.
        distant_list.append(np.linalg.norm(new_centroid - centroid))

        # Update centroid
        centroid_arr[i] = new_centroid

    mean_distance = np.mean(distant_list)
    plt.title("iteration %s, Updated distance=%.4f" % (iteration, mean_distance))
    plt.draw()
    plt.pause(0.5)
    ax.clear()
    # We assume converged when centroid no more updated that same as k-means.
    if mean_distance < 0.0001:
        break
    iteration += 1

# Sort all centroid(points) alone x then y value
ind = np.lexsort((centroid_arr[:, 1], centroid_arr[:, 0]))
sorted_centroids = centroid_arr[ind]

# If distance between points greater than threshold we split sorted centroids from those position to make cluster
centroid_diff = np.linalg.norm(np.diff(sorted_centroids, axis=0), axis=1)
split_idx = np.argwhere(centroid_diff > 1).ravel()
clustered_centroid = np.split(sorted_centroids, split_idx)

# Combine with k-means algorithm
new_centroids = []
for i, centroids in enumerate(clustered_centroid):
    new_centroid = np.mean(centroids, axis=0)
    ax.scatter(*new_centroid, color="C%d" % i, marker="*", s=200, alpha=1.0)
    new_centroids.append(new_centroid)

for point in all_points:
    distant_per_centroid = []
    # Calculate distance per point with each centroid.
    for cp in new_centroids:
        distant_per_centroid.append(np.linalg.norm(cp - point))
    # Get minimal distance between each centroid and each point, and choose the centroid point.
    cluster_idx = np.argmin(distant_per_centroid)
    ax.scatter(*point, color="C%d" % cluster_idx, s=50, alpha=0.1)

plt.title("Clustering result: %s cluster" % (i+1))
# Show end plot forever
plt.show()

