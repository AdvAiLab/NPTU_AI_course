import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

data_dim = 2
goal_num = 5
chromosome_num = 10

goals = np.random.rand(goal_num, data_dim)
start_point = np.random.rand(data_dim)
starts = np.repeat(start_point[np.newaxis], chromosome_num, axis=0)

temp = np.arange(goal_num)[np.newaxis]
population = np.repeat(temp, chromosome_num, axis=0)
list(map(np.random.shuffle, population))
starts = starts.reshape(10, 1, 2)
points = np.concatenate((starts, goals[np.newaxis, population][0]), axis=1)
print(points)


def plot_path(color='blue'):
    best_path_i = np.argmin(get_dist())
    for chromo in points:
        plt.plot(*chromo.T, c="grey")

    plt.plot(*points[best_path_i].T, c=color)


def get_dist():
    dist = points[:-1] - points[1:]
    dist_sum = np.linalg.norm(dist, axis=-1).sum()
    return dist_sum


plot_path(color='blue')

plt.scatter(*goals.T, c="red", s=250, alpha=0.7, marker="*")
plt.scatter(*start_point, c="green", s=250, alpha=0.7, marker="*")

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.grid()
plt.show()
