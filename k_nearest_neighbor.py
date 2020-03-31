import numpy as np
import matplotlib.pyplot as plt

plt.ion()

cluster_num = 100
cluster_dim = 2
cluster_shape = (cluster_num, cluster_dim)
cluster_colors = ["red", "blue", "green"]


def spawn_points():
    rand_points1 = 0 + 2 * np.random.rand(*cluster_shape)
    rand_points2 = 10 + 3 * np.random.rand(*cluster_shape)
    rand_points3 = [20, 0] + 2 * np.random.rand(*cluster_shape)
    points_num = cluster_num * 3
    points_list = [rand_points1, rand_points2, rand_points3]
    return points_num, points_list


points_num, points_list = spawn_points()
all_points = np.concatenate(points_list)

neighbors_K = 3

target_point = [3, 5]
distant_arr = np.zeros((points_num, neighbors_K))

for i, color in enumerate(zip(points_list, cluster_colors)):
    plt.scatter(*cp, color=color, marker='+', s=200)

    for j, ap in enumerate(all_points):
        distant_arr[j, i] = np.linalg.norm(cp - ap)

