import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from matplotlib.lines import Line2D
from test_figures.plot_figures_utils import get_figsize


def func_quad(x1, x2):
    f = x1**2 + x2**2
    return f


def draw_constraints_3d():
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 6})
    rcParams.update({'font.size': 7})

    lower_x_lim = -1
    upper_x_lim = 1
    l_thresh = 0.3
    u_thresh = 0.75

    num_points_for_grid_axis = 500  # Increase this to get a better looking figure (to 1000)
    num_contour_levels = 50  # Increase this to get a better looking figure (to 100)
    x1 = np.linspace(lower_x_lim, upper_x_lim, num_points_for_grid_axis)
    x2 = np.linspace(lower_x_lim, upper_x_lim, num_points_for_grid_axis)
    x1_grid, x2_grid = np.meshgrid(x1, x2)

    fig = plt.figure(figsize=get_figsize())
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1.2])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Quadratic')
    f = func_quad(x1_grid, x2_grid)

    f_l = f.copy()
    f_l[f < l_thresh] = np.nan
    levels = np.linspace(l_thresh, 1.0, num_contour_levels)
    ax.contour(x1, x2, f_l, alpha=0.1, colors='blue', levels=levels)
    projection = np.zeros_like(f_l)
    projection[f < l_thresh] = np.nan
    ax.contourf(x1, x2, projection, alpha=0.2, colors='blue', levels=1)

    f_u = f.copy()
    f_u[f > u_thresh] = np.nan
    levels = np.linspace(0, u_thresh, num_contour_levels)
    ax.contour(x1, x2, f_u, alpha=0.2, colors='orange', levels=levels)
    projection = np.zeros_like(f_l)
    projection[f > u_thresh] = np.nan
    ax.contourf(x1, x2, projection, alpha=0.2, colors='orange', levels=1)

    # Plot points x0, x
    x0 = (0.5, 0.5)
    f = func_quad(x0[0], x0[1])
    ax.scatter(x0[0], x0[1], f, marker='o', s=30, label='$f(x_0)$', color='red')
    ax.scatter(x0[0], x0[1], 0, marker='o', s=30, label='$x_0$', color='red', alpha=0.3)

    x = (-0.5, -0.5)
    f = func_quad(x[0], x[1])
    ax.scatter(x[0], x[1], f, marker='o', s=30, label='$f(x)$', color='green')
    ax.scatter(x[0], x[1], 0, marker='o', s=30, label='$x$', color='green', alpha=0.3)

    # Plot points x0_new
    x0_new = (0.0, 0.0)
    f = func_quad(x0_new[0], x0_new[1])
    ax.scatter(x0_new[0], x0_new[1], f, marker='o', s=30, label=r'$\bar{x}$', color='purple')

    ax.plot([x0[0], x0_new[0]], [x0[1], x0_new[1]], '--r', alpha=0.4)
    ax.scatter(x0_new[0]+0.05, x0_new[1]+0.05, f, marker='>', s=50, color='red', alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    colors = ['blue', 'orange']
    new_handles = [Line2D([0], [0], color=c, alpha=0.4, linewidth=3, linestyle='solid') for c in colors]
    new_labels = ['$L$ + proj', '$U$ + proj']
    handles += new_handles
    labels += new_labels
    ax.legend(handles, labels, ncol=4, loc="upper center")

    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    ax.view_init(15, 133)

    plt.subplots_adjust(top=0.99, bottom=0.05, left=0.01, right=0.95)
    fig_file_name = "quadratic_rlv_miss_violation.pdf"
    plt.savefig(fig_file_name)
    plt.close(fig)
    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    draw_constraints_3d()
