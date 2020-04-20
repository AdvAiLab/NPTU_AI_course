import numpy as np
import matplotlib.pyplot as plt
from util_3d import add_plot

is_3d = True
ax, particle_dim = add_plot(is_3d)

n_particles = 10
particles_shape = (n_particles, particle_dim)

test_goal = np.random.rand(particle_dim)

iteration = 0

best_goal = None
best_distant = None

while True:
    distant_list = []
    if best_goal is None:
        # First time we generate particles using Uniform Distribution
        rand_particles = np.random.rand(*particles_shape)
    else:
        # Randomly generate particles using Normal Distribution
        sigma = 1/(4*iteration)
        rand_particles = best_goal + sigma * np.random.randn(*particles_shape)

    ax.scatter(*rand_particles.T, s=50, alpha=0.5)
    ax.scatter(*test_goal, s=200, marker="*", alpha=1.0)

    for p in rand_particles:
        distance = np.linalg.norm(p - test_goal)
        distant_list.append(distance)
    min_idx = np.argmin(distant_list)
    min_particle = rand_particles[min_idx]

    if best_distant is None or distant_list[min_idx] < best_distant:
        best_distant = distant_list[min_idx]
        best_goal = min_particle
    ax.scatter(*best_goal, s=200, marker="+", alpha=1.0)

    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    if is_3d:
        ax.set_zlim3d(-1, 1)

    plt.title("iteration %s, Error: %.4f" % (iteration, best_distant))
    plt.pause(0.5)
    plt.draw()
    ax.clear()

    # We assume converged when centroid no more updated that same as k-means.
    # if mean_distance < 0.0001:
    #     break
    iteration += 1

# Show end plot forever
plt.show()
