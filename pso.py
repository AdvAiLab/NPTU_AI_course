import matplotlib.pyplot as plt
import numpy as np

from util_plot import AddPlot

is_3d = False
plots = AddPlot(is_3d)
ax, data_dim = plots.returns

# Make goal as gene
p_num = 3
p_arr = np.random.rand(p_num, data_dim)
v_arr = np.random.rand(p_num, data_dim)
pbest_arr = np.zeros((p_num, data_dim))
pbest_fit_arr = np.zeros(p_num)
gbest = None
gbest_fitness = 0.0

goal = np.random.rand(data_dim)

p_s = ax.scatter([], [])
pb_s = ax.scatter([], [], c="grey", alpha='0.5')
gb_s = ax.scatter([], [], c="green", marker="+", s=200)
goal_s = ax.scatter([], [], c="orange", marker="*", s=200)
arrow = ax.quiver(*p_arr.T, *v_arr.T, scale_units='xy', angles='xy', scale=1, color="red", alpha=0.5)

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
if is_3d:
    ax.set_zlim(-0.1, 1.1)
plt.grid()

# Parameters
inertia_weight = 0.7298
const_vp = 1.49618
const_vg = 1.49618
early_stop_fitness = 0.99

i = 0
iter_num = 100
while True:
    # Get fitness from distance
    dist_sum = np.linalg.norm(goal - p_arr, axis=-1)
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

    # Plot
    p_s.set_offsets(p_arr)
    pb_s.set_offsets(pbest_arr)
    gb_s.set_offsets(gbest)
    goal_s.set_offsets(goal)

    # We assume converged when arrive early_stop_fitness.
    if gbest_fitness > early_stop_fitness:
        ax.set_title("Stop at iteration %s, best_fitness: %.4f" % (i, gbest_fitness))
        break
    else:
        ax.set_title("iteration %s, best_fitness: %.4f" % (i, gbest_fitness))

    # PSO Algorithm
    rand1 = np.random.rand(p_num, data_dim)
    rand2 = np.random.rand(p_num, data_dim)
    v_arr[:] = (inertia_weight * v_arr) + \
               (rand1 * const_vp * (pbest_arr - p_arr)) + \
               (rand2 * const_vg * (gbest - p_arr))
    arrow.set_offsets(p_arr)
    arrow.set_UVC(*v_arr.T)
    p_arr[:] = p_arr + v_arr
    plt.draw()
    plt.pause(0.1)

    i += 1
plt.show()
