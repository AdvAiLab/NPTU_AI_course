import numpy as np
import matplotlib.pyplot as plt

n_particles = 20
particle_dim = 2
particles_shape = (n_particles, particle_dim)

# Randomly generate clusters using Normal Distribution (randn)
rand_particles = np.random.rand(*particles_shape)

test_goal = [0.5, 0.75]

iteration = 0

best_goal = None
best_distant = None

while True:
    distant_list = []
    rand_particles = np.random.rand(*particles_shape)
    plt.scatter(rand_particles[:, 0], rand_particles[:, 1], s=50, alpha=0.5)
    plt.scatter(*test_goal, s=200, marker="*", alpha=1.0)

    for p in rand_particles:
        distance = np.linalg.norm(p - test_goal)
        distant_list.append(distance)
    min_idx = np.argmin(distant_list)
    min_particle = rand_particles[min_idx]
    if best_distant is None:
        best_distant = distant_list[min_idx]
        best_goal = min_particle

    if distant_list[min_idx] < best_distant:
        best_distant = distant_list[min_idx]
        best_goal = min_particle
    plt.scatter(*best_goal, s=200, marker="+", alpha=1.0)

    plt.ylim((0, 1.0))
    plt.xlim((0, 1.0))

    plt.title("iteration %s" % iteration)
    plt.pause(0.5)
    plt.draw()
    plt.clf()

    # We assume converged when centroid no more updated that same as k-means.
    # if mean_distance < 0.0001:
    #     break
    iteration += 1

plt.show()
