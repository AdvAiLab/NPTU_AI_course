import matplotlib.pyplot as plt
import numpy as np
from util_plot import AddPlot

p = AddPlot(is_3d=True, with_lc=True)
ax1, _ = p.returns

x_min = -4
x_max = 4
y_min = -4
y_max = 4
z_min = 0
z_max = 13

# Add step to make more smooth and more slow
x = np.arange(x_min, x_max, 0.5)
y = np.arange(y_min, y_max, 0.5)
xx, yy = np.meshgrid(x, y, sparse=True)


def reversed_ackley_function(x, y):
    z = 15 - (-20 * np.exp(-0.2 * (np.sqrt(0.5 * (x ** 2 + y ** 2)))) -
              np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20)
    return z


zz = reversed_ackley_function(xx, yy)


def plot_ackley():
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='terrain', alpha=0.2)


chromosome_num = 20
gene_num = 2
population_shape = (chromosome_num, gene_num)

# Highest point on Reversed Ackley surface
goal_idx = np.unravel_index(np.argmax(zz, axis=None), zz.shape)
test_goal = np.array((x[goal_idx[0]], y[goal_idx[1]], zz[goal_idx]))

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
old_fitness = 0.0
early_stop_fitness = 0.98

# First time we generate population using Uniform Distribution
rand_population = np.random.rand(chromosome_num, gene_num) * [x_max - x_min, y_max - y_min] + \
                  [x_min, y_min]
# Add additional zero column
population = np.zeros((chromosome_num, gene_num + 1))
population[:, :-1] = rand_population

while True:
    plot_ackley()

    # Calculate fitness
    fitness = reversed_ackley_function(*population[:, :2].T)
    population[:, 2] = fitness

    # Normalize all to between 0 to 1
    fitness_arr = fitness / zz[goal_idx]

    # Get best goal
    max_idx = np.argmax(fitness_arr)
    max_chromosome = population[max_idx]

    if fitness_arr[max_idx] > best_fitness:
        best_fitness = fitness_arr[max_idx]
        best_goal = np.copy(max_chromosome)
        print(best_goal)

    # Plot
    ax1.scatter(*population.T, s=50, alpha=0.5)
    ax1.scatter(*test_goal, s=200, marker="*", alpha=1.0)
    ax1.scatter(*best_goal, s=200, marker="+", alpha=1.0)
    p.plot_curve(iteration, best_fitness)
    # We assume converged when arrive early_stop_fitness.
    if best_fitness > early_stop_fitness:
        ax1.set_title("Stop at iteration %s, best_fitness: %.4f" % (iteration, best_fitness))
        break
    else:
        ax1.set_title("iteration %s, best_fitness: %.4f" % (iteration, best_fitness))

    plt.draw()
    plt.pause(0.5)
    # Clear surface plot
    ax1.clear()

    # Genetic Algorithm

    # Selection
    sorted_idx = np.argsort(fitness_arr)
    sel_chromo = population[sorted_idx][-selection_num:]
    # Copy selected chromosomes randomly to fulfill original length of population
    copy_idx = np.random.choice(selection_num, copy_num)
    copy_pop = sel_chromo[copy_idx]
    population = np.concatenate((sel_chromo, copy_pop))

    # Crossover
    pick_idx = (np.random.rand(chromosome_num) <= crossover_rate).nonzero()[0]
    parent_idx = np.random.randint(chromosome_num, size=pick_idx.shape[0])
    rand_index = np.random.randint(gene_num, size=pick_idx.shape[0])
    # Swap
    population[[parent_idx, pick_idx], rand_index] = population[[pick_idx, parent_idx], rand_index]

    # Mutation
    pick_idx = (np.random.rand(chromosome_num) <= mutation_rate).nonzero()[0]
    rand_index = np.random.randint(gene_num, size=pick_idx.shape[0])
    population[pick_idx, rand_index] = np.random.normal(best_goal[rand_index])

    iteration += 1
# Show end plot forever
plt.show()
