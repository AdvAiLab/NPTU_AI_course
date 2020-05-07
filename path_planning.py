import matplotlib.pyplot as plt
import numpy as np

from util_3d import add_plot

is_3d = True
ax, data_dim = add_plot(is_3d)

# Make goal as gene
gene_num = 5
chromosome_num = 10

# Create points and paths
points = np.random.rand(1+gene_num, data_dim)
paths = np.zeros((chromosome_num, *points.shape)) + points

population = np.zeros((chromosome_num, gene_num), dtype=int) + np.arange(gene_num)
list(map(np.random.shuffle, population))

# Parameters
mutation_rate = 0.3
selection_ratio = 0.7
selection_num = int(chromosome_num * selection_ratio)
copy_num = chromosome_num - selection_num

best_fitness = 0.0
early_stop_fitness = 1.0

iteration = 0
iter_num = 100
plt.pause(3)
while True:
    # Update all paths
    paths[:, 1:] = points[1:][population]
    # Get fitness from distance
    diff = np.diff(paths, axis=1)
    dist_sum = np.linalg.norm(diff, axis=-1).sum(axis=1)
    fitness_list = 1 / (1 + dist_sum)
    best_path_i = np.argmax(fitness_list)
    if fitness_list[best_path_i] > best_fitness:
        best_fitness = fitness_list[best_path_i]
        best_chromosome = np.copy(paths[best_path_i])

    # Plot path
    ax.scatter(*points[0], c="green", s=250, alpha=0.7, marker="*")
    ax.scatter(*points[1:].T, c="red", s=250, alpha=0.7, marker="*")
    for chromo in paths:
        ax.plot(*chromo.T, c="grey")
    ax.plot(*best_chromosome.T, c="blue")

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    if is_3d:
        ax.set_zlim(-0.1, 1.1)
    plt.grid()

    # We assume converged when arrive early_stop_fitness.
    if best_fitness > early_stop_fitness or iteration >= iter_num - 1:
        plt.title("Stop at iteration %s, best_fitness: %.4f" % (iteration, best_fitness))
        break
    else:
        plt.title("iteration %s, best_fitness: %.4f" % (iteration, best_fitness))
    plt.draw()
    plt.pause(0.5)
    ax.clear()

    # Genetic Algorithm

    # Selection
    sorted_idx = np.argsort(fitness_list)
    sel_chromo = population[sorted_idx][-selection_num:]
    # Copy selected chromosomes randomly
    copy_pop = sel_chromo[np.random.choice(selection_num, copy_num)]
    # Append all to original length of population
    population = np.concatenate((sel_chromo, copy_pop))

    # Mutation
    rand_chromo = np.argwhere(np.random.rand(chromosome_num) <= mutation_rate)
    # Swap
    for i in rand_chromo:
        rand_gene = np.random.choice(gene_num, 2, replace=False)
        population[i, rand_gene] = population[i, rand_gene[::-1]]
    iteration += 1
plt.show()
