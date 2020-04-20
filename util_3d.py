from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def add_plot(is_3d):
    fig = plt.figure()
    # mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    # mngr.window.wm_geometry("+350+100")
    if is_3d:
        point_dim = 3
        ax = fig.add_subplot(111, projection='3d')
    else:
        point_dim = 2
        ax = fig.add_subplot(111)
    return ax, point_dim
