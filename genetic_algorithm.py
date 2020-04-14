import numpy as np
import matplotlib.pyplot as plt

chromosome_num = 10
gene_num = 2
population_shape = (chromosome_num, gene_num)

test_goal = np.random.rand(gene_num)

iteration = 0
iteration_num = 100
mutation_rate = 0.3
crossover_rate = 0.3
selection_ratio = 0.3

selection_num = int(chromosome_num*selection_ratio)
copy_num = chromosome_num - selection_num

best_goal = None
best_fitness = 0.0
early_stop_fitness = 0.99

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

    # We assume converged when arrive early_stop_fitness.
    if best_fitness > early_stop_fitness:
        break
    plt.clf()

    # Plot
    plt.scatter(population[:, 0], population[:, 1], s=50, alpha=0.5)
    plt.scatter(*test_goal, s=200, marker="*", alpha=1.0)
    plt.scatter(*best_goal, s=200, marker="+", alpha=1.0)

    plt.ylim(0, 1)
    plt.xlim(0, 1)

    plt.title("iteration %s, best_fitness: %.4f" % (iteration, best_fitness))
    plt.pause(0.5)
    plt.draw()

    # Genetic Algorithm

    # Selection
    sorted_idx = np.argsort(fitness_list)
    selected_pop = population[sorted_idx][-selection_num:]
    copy_idx = np.random.choice(selected_pop.shape[0], copy_num)
    copy_pop = selected_pop[copy_idx]
    population = np.concatenate((selected_pop, copy_pop))

    # Crossover
    for i, chromosome in enumerate(population):
        if np.random.rand(1) <= mutation_rate:
            # Prevent to crossover with self
            parent_idx = np.random.randint(population.shape[0] - 1)
            if parent_idx >= i:
                parent_idx += 1

            rand_index = np.random.choice(chromosome.shape[0], 1)
            # Swap
            buff_gene = population[parent_idx][rand_index]
            population[parent_idx][rand_index] = population[i][rand_index]
            population[i][rand_index] = buff_gene

    # Mutation
    for i, chromosome in enumerate(population):
        if np.random.rand(1) <= mutation_rate:
            rand_index = np.random.randint(chromosome.shape[0])
            population[i][rand_index] = np.random.rand(1)

    iteration += 1

plt.show()
