import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from util_plot import AddPlot

is_3d = True
ax, point_dim = AddPlot(is_3d).returns

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :point_dim]  # we only take the first point_dim features.
y = iris.target

if is_3d:
    test_point = np.array([7, 4, 3])
else:
    test_point = np.array([7, 4])

# Set number of neighbors and (x, y) of test point.
neighbors_K = 3

distant_arr = np.zeros(len(y))

for i, (point, color_idx) in enumerate(zip(X, y)):
    # Plot all points
    ax.scatter(*point, color="C%d" % color_idx, s=50, alpha=0.1)
    distant_arr[i] = np.linalg.norm(test_point - point)

# Get neighbor points from sorted distance
min_idx = np.argsort(distant_arr)[:neighbors_K]
neighbor_points = X[min_idx]
neighbor_colors_idx = y[min_idx]

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
    ax.scatter(*test_point, color="black", marker="*", s=200)
    print("You got multiple predicted colors: %s" % results)

plt.show()
