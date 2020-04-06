import matplotlib.pyplot as plt
import numpy as np

n_cluster_points = 20
point_dim = 2
cluster_shape = (n_cluster_points, point_dim)
cluster_colors = ["red", "blue", "green"]
color_per_point = []

# Set number of neighbors and (x, y) of test point.
neighbors_K = 5
test_point = np.array([2, 5])

# Spawn points
rand_points1 = 0 + 2 * np.random.randn(*cluster_shape)
rand_points2 = 7 + 3 * np.random.randn(*cluster_shape)
rand_points3 = [3, 0] + 2 * np.random.randn(*cluster_shape)
points_num = n_cluster_points * 3
points_list = [rand_points1, rand_points2, rand_points3]
distant_arr = np.zeros(points_num)

for i, (cluster, color) in enumerate(zip(points_list, cluster_colors)):
    # Plot all points
    plt.scatter(cluster[:, 0], cluster[:, 1], color=color, s=50, alpha=0.1)
    # Create color for each point
    color_per_point.append(np.full(n_cluster_points, i, dtype=int))
color_per_point = np.concatenate(color_per_point)
all_points = np.concatenate(points_list)

# Calculate distance between test point and each point
for i, ap in enumerate(all_points):
    distant_arr[i] = np.linalg.norm(test_point - ap)
# Find neighbor points which has minimal distance
min_idx = np.argsort(distant_arr)
neighbor_points = all_points[min_idx][:neighbors_K]
neighbor_colors = color_per_point[min_idx][:neighbors_K]

# Emphasize neighbor points
for p, color in zip(neighbor_points, neighbor_colors):
    plt.scatter(*p, color=cluster_colors[color], s=50, alpha=0.5)

# Get value of maximum count
u, c = np.unique(neighbor_colors, return_counts=True)
y = u[c == c.max()]
results = [cluster_colors[c] for c in y]
if len(y) == 1:
    plt.scatter(*test_point, color=results[0], marker="*", s=200)
else:
    raise AssertionError("You got multiple predicted result: %s" % results)

plt.show()
