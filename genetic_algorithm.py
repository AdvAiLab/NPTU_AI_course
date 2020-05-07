import numpy as np
import matplotlib.pyplot as plt
from util_plot import AddPlot

is_3d = True
ax, gene_num = AddPlot(is_3d).returns

chromosome_num = 10
population_shape = (chromosome_num, gene_num)

test_goal = np.random.rand(gene_num)

iteration = 0
iteration_num = 100

# Parameters
mutation_rate = 0.3
crossover_rate = 0.3
selection_ratio = 0.3

selection_num = int(chromosome_num * selection_ratio)
copy_num = chromosome_num - selection_num

best_goal = None
best_fitness = 0.0
early_stop_fitness = 0.98

# First time we generate population using Uniform Distribution
population = np.random.rand(*population_shape)

while True:
    # Calculate fitness
    fitness_list = []
    for chromosome in population:
        fitness = 1.0 / (1.0 + np.linalg.norm(chromosome - test_goal))
        fitness_list.append(fitness)

    # Get best goal
    max_idx = np.argmax(fitness_list)
    max_chromosome = population[max_idx]

    if fitness_list[max_idx] > best_fitness:
        best_fitness = fitness_list[max_idx]
        best_goal = np.copy(max_chromosome)
        print(best_goal)


    # Plot
    ax.scatter(*population.T, s=50, alpha=0.5)
    ax.scatter(*test_goal, s=200, marker="*", alpha=1.0)
    ax.scatter(*best_goal, s=200, marker="+", alpha=1.0)

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    if is_3d:
        ax.set_zlim3d(0, 1)

    plt.title("iteration %s, best_fitness: %.4f" % (iteration, best_fitness))
    plt.draw()
    plt.pause(0.5)
    # We assume converged when arrive early_stop_fitness.
    if best_fitness > early_stop_fitness:
        plt.title("Stop at iteration %s, best_fitness: %.4f" % (iteration, best_fitness))
        break
    ax.clear()

    # Genetic Algorithm

    # Selection
    sorted_idx = np.argsort(fitness_list)
    selected_chromosomes = population[sorted_idx][-selection_num:]
    # Copy selected chromosomes randomly to fulfill original length of population
    copy_idx = np.random.choice(selected_chromosomes.shape[0], copy_num)
    copy_pop = selected_chromosomes[copy_idx]
    population = np.concatenate((selected_chromosomes, copy_pop))

    # Crossover
    for i in range(chromosome_num):
        if np.random.rand(1) <= crossover_rate:
            # Prevent to crossover with self
            parent_idx = np.random.randint(population.shape[0] - 1)
            if parent_idx >= i:
                parent_idx += 1

            rand_index = np.random.choice(gene_num, 1)
            # Swap
            buff_gene = population[parent_idx][rand_index]
            population[parent_idx][rand_index] = population[i][rand_index]
            population[i][rand_index] = buff_gene

    # Mutation
    for i in range(chromosome_num):
        if np.random.rand(1) <= mutation_rate:
            rand_index = np.random.randint(gene_num)
            population[i][rand_index] = np.random.rand(1)

    iteration += 1
# Show end plot forever
plt.show()
