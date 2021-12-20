import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from matplotlib import cm
from utils.test_utils import read_config_file
from utils.data_generator import DataGeneratorQuadraticInverse
from tests.visualization.utils import get_figsize
from utils.functions_to_monitor import func_quadratic_inverse

low = -2
high = 2


def plot_surface(x1_grid, x2_grid, f):
    fig = plt.figure(figsize=get_figsize(wf=0.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([low, high])
    ax.set_ylim([low, high])
    ax.set_xlabel('x\u2081', labelpad=-8)  # \u2081 is the unicode of subscript one: '$x_1$'
    ax.set_ylabel('x\u2082', labelpad=-8)  # \u2081 is the unicode of subscript one: '$x_2$'
    ax.set_zlabel('f', labelpad=-8)
    ax.plot_surface(x1_grid, x2_grid, f, cmap=cm.coolwarm.reversed())
    ax.set_zticks([-3, 0, 3])
    ax.tick_params(axis='x', which='major', pad=-5)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=-3)
    plt.subplots_adjust(top=1.1, bottom=0.1, left=0.00, right=0.9)
    return fig


def draw_f(test_folder):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 6})
    rcParams.update({'font.size': 7})

    num_points_for_grid_axis = 50  # Increase this to get a better looking figure
    x1 = np.linspace(low, high, num_points_for_grid_axis)
    x2 = np.linspace(low, high, num_points_for_grid_axis)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_ = np.append(x1_grid.reshape(x1_grid.shape[0] ** 2, -1), x2_grid.reshape(x2_grid.shape[0] ** 2, -1), axis=1)
    f = func_quadratic_inverse(x_)
    f = f.reshape(x1_grid.shape[0], -1)

    fig_f = plot_surface(x1_grid, x2_grid, f)

    fig_f.savefig(test_folder + "/quadratic_inverse_func_surface.pdf")
    plt.close(fig_f)
    rcParams.update(rcParamsDefault)


def draw_f_contour_and_node_trail(nodes_data, test_folder):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    fig = plt.figure(figsize=get_figsize(wf=0.5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('x\u2081')  # \u2081 is the unicode of subscript one: '$x_1$'
    ax.set_ylabel('x\u2082')  # \u2081 is the unicode of subscript one: '$x_2$'
    x1 = np.linspace(low, high, 50)
    x2 = np.linspace(low, high, 50)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_ = np.append(x1_grid.reshape(x1_grid.shape[0] ** 2, -1), x2_grid.reshape(x2_grid.shape[0] ** 2, -1), axis=1)
    f_values = func_quadratic_inverse(x_)
    f_values = f_values.reshape(x1_grid.shape[0], -1)
    plt.contourf(x1_grid, x2_grid, f_values, 20, cmap='RdBu')
    plt.colorbar(ticks=[-3, -2, -1, 0, 1, 2, 3])

    colors = ['tab:green', 'tab:green', 'green', 'green']
    markers = ["o", "o", "x", "x"]

    num_nodes = 4
    for i in range(num_nodes):
        node_data = nodes_data[i::4]
        node_start = np.mean(node_data[:20], axis=0)
        node_end = np.mean(node_data[-20:], axis=0)

        plt.scatter(node_data[50:-50:10, 0], node_data[50:-50:10, 1], s=0.05, color=colors[i], marker=markers[i])
        plt.arrow(node_start[0], node_start[1], node_end[0] - node_start[0], node_end[1] - node_start[1],
                  color='black', head_width=0.2, head_length=0.2, linewidth=0.01)

    ax.spines['right'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    plt.subplots_adjust(top=0.96, bottom=0.29, left=0.22, right=0.93)
    fig.savefig(test_folder + "/quadratic_inverse_func_contour_and_node_trail.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    data_folder = "../test_results/results_ablation_study_quadratic_inverse_2021-07-08_15-40-37"
    conf = read_config_file(data_folder)
    data_generator = DataGeneratorQuadraticInverse(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name="data_file.txt", test_folder=data_folder, d=conf["d"], sliding_window_size=conf["sliding_window_size"])
    draw_f_contour_and_node_trail(data_generator.data, "./")
    draw_f("./")
