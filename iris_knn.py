import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

from util_3d import add_plot

is_3d = False
ax, point_dim = add_plot(is_3d)

# Set number of neighbors and (x, y) of test point.
neighbors_K = 3

total_points_num = n_cluster_points * 3
points_list = [rand_points1, rand_points2, rand_points3]
distant_arr = np.zeros(total_points_num)

for i, cluster in enumerate(points_list):
    # Plot all points
    ax.scatter(*cluster.T, color="C%d" % i, s=50, alpha=0.1)
    # Create color for each point
    points_color_idx.append(np.full(n_cluster_points, i, dtype=int))

# Make list to concatenated np array alone axis 0
points_color_idx = np.concatenate(points_color_idx)
all_points = np.concatenate(points_list)

# Calculate distance between test point and each point
for i, ap in enumerate(all_points):
    distant_arr[i] = np.linalg.norm(test_point - ap)

# Get neighbor points from sorted distance
min_idx = np.argsort(distant_arr)[:neighbors_K]
neighbor_points = all_points[min_idx]
neighbor_colors_idx = points_color_idx[min_idx]

# Emphasize neighbor points
for p, color_idx in zip(neighbor_points, neighbor_colors_idx):
    ax.scatter(*p, color="C%d" % color_idx, s=50, alpha=0.5)

# Get value of unique item of maximum count
u, c = np.unique(neighbor_colors_idx, return_counts=True)
y = u[c == c.max()]
results = ["C%d" % c for c in y]

# Assert to only one predicted result of test point
if len(results) == 1:
    ax.scatter(*test_point, color=results[0], marker="*", s=200)
else:
    raise AssertionError("You got multiple predicted result: %s" % results)

plt.show()
