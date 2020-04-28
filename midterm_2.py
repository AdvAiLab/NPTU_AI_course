import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 8))
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212)
ax2.set_title("Learning Curve")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Fitness Value")
ax2.grid()

x_min = -5
x_max = 5
y_min = -5
y_max = 5
z_min = 0
z_max = 80

chromo_num = 20
gene_num = 2

sel_rate = 0.3
crossover_rate = 0.3
mutation_rate = 0.3

sel_num = int(chromo_num * sel_rate)
copy_num = chromo_num - sel_num


def reversed_rastrigin_function(x, y):
    A = 10
    z = 80 - (2 * A + x ** 2 + y ** 2 - A * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return z


x = np.arange(x_min, x_max, 0.1)
y = np.arange(y_min, y_max, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)

zz = reversed_rastrigin_function(xx, yy)


def plot_rastrigin():
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='terrain', alpha=0.2)


rand_population = np.random.rand(chromo_num, gene_num) * [x_max - x_min, y_max - y_min] + [x_min, y_min]
# Add addition zero column
population = np.zeros((chromo_num, gene_num + 1))
population[:, :-1] = rand_population

best_fitness = 0.0
early_stop_fitness = 79
iteration = 0
max_iteration = 30
while True:
    ax1.clear()

    plot_rastrigin()

    # Get fitness as z
    population[:, 2] = reversed_rastrigin_function(*population[:, :2].T)
    fitness_arr = population[:, 2]

    max_idx = np.argmax(fitness_arr)
    old_fitness = best_fitness

    if fitness_arr[max_idx] > best_fitness:
        best_fitness = fitness_arr[max_idx]
        best_goal = np.copy(population[max_idx])

    ax1.scatter(*population.T, s=50, c='red')
    ax1.scatter(*best_goal, s=200, c='blue', marker='*')

    if iteration > 0:
        ax2.plot((iteration - 1, iteration), (old_fitness, best_fitness), c='C0')

    if best_fitness < early_stop_fitness:
        ax1.set_title("iteration %d, fitness %.4f" % (iteration, best_fitness))
    elif iteration >= max_iteration-1:
        ax1.set_title("Early Stop at iteration %d, fitness %.4f" % (iteration, best_fitness))
        break
    else:
        ax1.set_title("Stop at iteration %d, fitness %.4f" % (iteration, best_fitness))
        break

    plt.draw()
    plt.pause(0.5)

    # Genetic Algorithm

    # Selection
    sorted_idx = np.argsort(fitness_arr)
    seled_chromo = population[sorted_idx][-sel_num:]
    copy_idx = np.random.choice(seled_chromo.shape[0], copy_num)
    copy_chromo = seled_chromo[copy_idx]
    population = np.concatenate((seled_chromo, copy_chromo))

    # Crossover
    for i in range(chromo_num):
        if np.random.rand(1) <= crossover_rate:
            gene_idx = np.random.randint(gene_num)
            # Prevent crossover with self
            parent_idx = np.random.randint(chromo_num - 1)
            if i == parent_idx:
                parent_idx = chromo_num - 1

            buff_gene = population[parent_idx, gene_idx]
            population[parent_idx, gene_idx] = population[i, gene_idx]
            population[i, gene_idx] = buff_gene

    # Mutation
    for i in range(chromo_num):
        if np.random.rand(1) <= mutation_rate:
            gene_idx = np.random.randint(gene_num)
            population[i, gene_idx] = np.random.normal(best_goal[gene_idx]) / (iteration+1)

    iteration += 1

# Plot forever after finished
plt.draw()
plt.show()
