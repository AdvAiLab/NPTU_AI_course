import numpy as np
import matplotlib.pyplot as plt

chromosome_num = 10
gene_num = 2
population_shape = (chromosome_num, gene_num)

test_goal = np.random.rand(gene_num)

iteration = 0

best_goal = None
best_fitness = None

while True:
    fitness_list = []
    # First time we generate particles using Uniform Distribution
    population = np.random.rand(*population_shape)

    plt.scatter(population[:, 0], population[:, 1], s=50, alpha=0.5)
    plt.scatter(*test_goal, s=200, marker="*", alpha=1.0)

    for p in population:
        fitness = 1.0 / (1.0 + np.linalg.norm(p - test_goal))
        fitness_list.append(fitness)

    max_idx = np.argmax(fitness_list)
    max_particle = population[max_idx]
    if best_fitness is None:
        best_fitness = fitness_list[max_idx]
        best_goal = max_particle

    if fitness_list[max_idx] > best_fitness:
        best_fitness = fitness_list[max_idx]
        best_goal = max_particle
    plt.scatter(*best_goal, s=200, marker="+", alpha=1.0)

    plt.ylim(0, 1)
    plt.xlim(0, 1)

    plt.title("iteration %s, Error: %.4f" % (iteration, best_fitness))
    plt.pause(0.5)
    plt.draw()
    plt.clf()

    # We assume converged when centroid no more updated that same as k-means.
    # if mean_distance < 0.0001:
    #     break
    iteration += 1

plt.show()
