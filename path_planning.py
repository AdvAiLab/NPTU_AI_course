import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

data_dim = 2
goal_num = 5

goals = np.random.rand(goal_num, data_dim)
start = np.random.rand(data_dim)

temp = np.arange(goal_num)
np.random.shuffle(temp)


def plot_path(temp, color):
    points = np.concatenate((start[np.newaxis], goals[temp]))
    plt.plot(*points.T, c="blue")


plot_path(temp, color='blue')

plt.scatter(*goals.T, c="red", s=250, alpha=0.7, marker="*")
plt.scatter(*start, c="green", s=250, alpha=0.7, marker="*")

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.grid()
plt.show()
