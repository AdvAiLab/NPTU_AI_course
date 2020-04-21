import numpy as np

np.set_printoptions(threshold=np.inf)
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')


def plot_ackley():
    x_min = -6
    x_max = 6
    y_min = -6
    y_max = 6

    x = np.arange(x_min, x_max, 0.5)
    y = np.arange(y_min, y_max, 0.5)
    x, y = np.meshgrid(x, y)

    a = np.pi
    b = np.e

    z = 15 - (-20 * np.exp(-0.2 * (np.sqrt(0.5 * (x ** 2 + y ** 2)))) -
              np.exp(0.5 * (np.cos(2 * a * x) + np.cos(2 * a * y))) + b + 20)
    print(z)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(0, 13)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='terrain', alpha=0.2)
    plt.show()


plot_ackley()

gene_num = 3
chromosome_num = 20
iteration_num = 50
mutation_rate = 0.3
crossover_rate = 0.7
select_ratio = 0.3

select_num = int(chromosome_num * select_ratio)
copy_num = int(chromosome_num * select_num)

population = np.random.rand(chromosome_num, gene_num)
goal = np.random.rand(gene_num)
best_fitness = 0
best_chromosome = np.array([0, 0])

fitness_array = []

for i in range(chromosome_num):
    fitness = 1.0 / (1.0 + distance.euclidean(population[i, :], goal))
    fitness_array.append(fitness)

for iteration in range(iteration_num):

    temp_population = np.copy(population)
    selected_idx = np.argsort(fitness_array)[-select_num:]
    temp_population = np.copy(population[selected_idx])

    for i in range(copy_num):
        sel_chromosome = np.random.randint(0, select_num)
        copy_chromosome = np.copy(temp_population[sel_chromosome, :].reshape((1, gene_num)))
        temp_population = np.concatenate((temp_population, copy_chromosome), axis=0)

    population = np.copy(temp_population)

    for i in range(chromosome_num):
        if np.random.rand(1) < crossover_rate:
            sel_chromosome = np.random.randint(0, chromosome_num)
            sel_gene = np.random.randint(0, gene_num)

            temp = np.copy(population[i, sel_gene])
            population[i, sel_gene] = np.copy(population[sel_chromosome, sel_gene])
            population[sel_chromosome, sel_gene] = np.copy(temp)

    for i in range(chromosome_num):
        if np.random.rand(1) < mutation_rate:
            sel_gene = np.random.randint(0, gene_num)
            population[i, sel_gene] = np.random.rand(1)

    if np.max(fitness_array) > best_fitness:
        best_fitness = np.max(fitness_array)
        best_idx = np.argmax(fitness_array)
        best_chromosome = np.copy(population[best_idx])

    error = distance.euclidean(best_chromosome, goal)

    plt.clf()
    ax.scatter(population[:, 0], population[:, 1], color='b', s=50, alpha=0.3, marker='o')
    ax.scatter(goal[0], goal[1], color='r', s=250, alpha=1.0, marker='*')

    plt.title('iteration: ' + str(iteration) + ' Error: ' + str(error))
    ax.grid()
    plt.grid()
    plt.show()
    plt.pause(0.2)
