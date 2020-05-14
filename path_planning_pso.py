import matplotlib.pyplot as plt
import numpy as np

from util_plot import AddPlot

is_3d = False
plots = AddPlot(is_3d)
ax, data_dim = plots.returns

# Make goal as gene
gene_num = 5
chromosome_num = 10
p_arr = np.random.rand(chromosome_num, gene_num)
v_arr = np.random.rand(chromosome_num, gene_num)
pbest_arr = np.zeros((chromosome_num, gene_num))
pbest_fit_arr = np.zeros(chromosome_num)
gbest = None
gbest_path = None
gbest_fitness = 0.0

# Create points and paths
points = np.random.rand(1 + gene_num, data_dim)
paths = np.repeat([points], chromosome_num, axis=0)

# population = np.repeat([np.arange(gene_num)], chromosome_num, axis=0)
# list(map(np.random.shuffle, population))

# Parameters
inertia_weight = 0.7298
const_vp = 1.49618
const_vg = 1.49618
early_stop_fitness = 1.0

i = 0
iter_num = 100


def update(pos, vel, pb, gb):
    rand1 = np.random.rand()
    rand2 = np.random.rand()

    vel[:] = (inertia_weight * vel) + \
             (rand1 * const_vp * (pb - pos)) + \
             (rand2 * const_vg * (gb - pos))
    pos[:] = pos + vel


while True:

    # Embedding representation
    sorted_idx = np.argsort(p_arr)
    # Update all paths
    paths[:, 1:] = points[1:][sorted_idx]
    # Get fitness from distance
    diff = np.diff(paths, axis=1)
    dist_sum = np.linalg.norm(diff, axis=-1).sum(axis=1)
    fitness_list = 1 / (1 + dist_sum)
    # update personal best
    updated_p = fitness_list > pbest_fit_arr
    pbest_fit_arr[updated_p] = fitness_list[updated_p]
    pbest_arr[updated_p] = p_arr[updated_p]
    # check global best
    best_path_i = fitness_list.argmax()
    if fitness_list[best_path_i] > gbest_fitness:
        gbest_fitness = fitness_list[best_path_i]
        gbest = np.copy(p_arr[best_path_i])
        gbest_path = np.copy(paths[best_path_i])

    # Plot path
    ax.scatter(*points[0], c="green", s=250, alpha=0.7, marker="*")
    ax.scatter(*points[1:].T, c="red", s=250, alpha=0.7, marker="*")
    for chromo in paths:
        ax.plot(*chromo.T, c="grey")
    ax.plot(*gbest_path.T, c="blue")

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    if is_3d:
        ax.set_zlim(-0.1, 1.1)
    plt.grid()

    # We assume converged when arrive early_stop_fitness.
    if gbest_fitness > early_stop_fitness or i >= iter_num - 1:
        plt.title("Stop at iteration %s, best_fitness: %.4f" % (i, gbest_fitness))
        break
    else:
        plt.title("iteration %s, best_fitness: %.4f" % (i, gbest_fitness))
    plt.draw()
    plt.pause(0.5)
    ax.clear()

    # PSO Algorithm
    update(p_arr, v_arr, pbest_arr, gbest)

    i += 1
plt.show()
