from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class AddPlot:
    def __init__(self, is_3d, pos=111, with_lc=False, is_record=False):
        if with_lc:
            pos = 211
            self.fig = plt.figure(figsize=(6, 8))
            self._add_learning_curve()
        else:
            self.fig = plt.figure()
        if is_record:
            mngr = plt.get_current_fig_manager()
            # to put it into the upper left corner for example:
            mngr.window.wm_geometry("+350+100")
        if is_3d:
            point_dim = 3
            ax = self.fig.add_subplot(pos, projection='3d')
        else:
            point_dim = 2
            ax = self.fig.add_subplot(pos)
        self.returns = ax, point_dim

    def _add_learning_curve(self):
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Learning Curve")
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Fitness Value")
        self.ax2.grid()
        self.prev_fitness = None

    def plot_curve(self, iteration, best_fitness):
        if self.prev_fitness is not None:
            self.ax2.plot((iteration - 1, iteration), (self.prev_fitness, best_fitness), c='C0')
        self.prev_fitness = best_fitness

