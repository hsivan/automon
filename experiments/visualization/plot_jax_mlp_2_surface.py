import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from matplotlib import cm
from test_utils.test_utils import read_config_file
from test_utils.jax_mlp import get_net_arch_large_dim
from test_utils.data_generator import DataGeneratorMlp
from experiments.visualization.visualization_utils import get_figsize
from jax.config import config
config.update("jax_platform_name", 'cpu')


def func_to_approx(x):
    y = x[:, 0] * jnp.exp(-jnp.sum(x ** 2, axis=1) / (x.shape[1] - 1))
    return y


def plot_surface(x1_grid, x2_grid, f, low, high):
    fig = plt.figure(figsize=get_figsize(wf=0.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([low, high])
    ax.set_ylim([low, high])
    ax.set_xlabel('x\u2081', labelpad=-10)  # \u2081 is the unicode of subscript one: '$x_1$'
    ax.set_ylabel('x\u2082', labelpad=-9)  # \u2082 is the unicode of subscript one: '$x_2$'
    ax.plot_surface(x1_grid, x2_grid, f, cmap=cm.coolwarm.reversed())
    ax.set_zticks([-0.3, 0, 0.3])
    ax.tick_params(axis='x', which='major', pad=-5)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=-3)

    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_linewidth(0.5)
    ax.tick_params(width=0.5)

    plt.subplots_adjust(top=1.1, bottom=0.1, left=0.00, right=0.9)
    return fig


def draw_f_and_f_approx(net_apply, network_params, low, high):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    num_points_for_grid_axis = 50  # Increase this to get a better looking figure
    x1 = jnp.linspace(low, high, num_points_for_grid_axis)
    x2 = jnp.linspace(low, high, num_points_for_grid_axis)
    x1_grid, x2_grid = jnp.meshgrid(x1, x2)
    x_ = jnp.append(x1_grid.reshape(x1_grid.shape[0] ** 2, -1), x2_grid.reshape(x2_grid.shape[0] ** 2, -1), axis=1)
    f = func_to_approx(x_)
    f = f.reshape(x1_grid.shape[0], -1)
    # Apply network to inputs
    f_predictions = net_apply(network_params, x_)
    f_predictions = f_predictions.reshape(x1_grid.shape[0], -1)

    fig_f = plot_surface(x1_grid, x2_grid, f, low, high)  # f = x_1 exp(-x_1^2 - x_2^2)
    fig_f_approx = plot_surface(x1_grid, x2_grid, f_predictions, low, high)  # f = trained network

    fig_f_approx.savefig("./mlp_2_surface.pdf")

    plt.close(fig_f)
    plt.close(fig_f_approx)

    rcParams.update(rcParamsDefault)


def draw_f_approx_contour_and_node_trail(net_apply, network_params, nodes_data, low, high):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    node1_trail = nodes_data[::10]  # trail of first node
    node6_trail = nodes_data[5::10]  # trail of sixth node

    # Average 20 first and last samples
    node1_start = jnp.mean(node1_trail[:20], axis=0)
    node1_end = jnp.mean(node1_trail[-20:], axis=0)
    node6_start = jnp.mean(node6_trail[:20], axis=0)
    node6_end = jnp.mean(node6_trail[-20:], axis=0)

    fig = plt.figure(figsize=get_figsize(wf=0.5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('x\u2081', labelpad=-1)  # \u2081 is the unicode of subscript one: '$x_1$'
    ax.set_ylabel('x\u2082', labelpad=-3)  # \u2082 is the unicode of subscript two: '$x_2$'
    x1 = jnp.linspace(low-0.5, high+0.8, 50)
    x2 = jnp.linspace(low-0.5, high+0.8, 50)
    x1_grid, x2_grid = jnp.meshgrid(x1, x2)
    x_ = jnp.append(x1_grid.reshape(x1_grid.shape[0] ** 2, -1), x2_grid.reshape(x2_grid.shape[0] ** 2, -1), axis=1)
    f_predictions = net_apply(network_params, x_)
    f_predictions = f_predictions.reshape(x1_grid.shape[0], -1)
    plt.contourf(x1_grid, x2_grid, f_predictions, 20, cmap='RdBu')
    plt.colorbar(ticks=[-0.3, 0, 0.3])
    plt.scatter(node1_trail[:, 0], node1_trail[:, 1], s=0.05, color='tab:green', marker="x")
    plt.scatter(node6_trail[:, 0], node6_trail[:, 1], s=0.05, color='tab:green', marker=".")
    plt.arrow(node1_start[0], node1_start[1], node1_end[0]-node1_start[0], node1_end[1]-node1_start[1], color='black', head_width=0.2, head_length=0.25)
    plt.arrow(node6_start[0], node6_start[1], node6_end[0] - node6_start[0], node6_end[1] - node6_start[1], color='black', head_width=0.2, head_length=0.25)

    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    plt.subplots_adjust(top=0.98, bottom=0.24, left=0.17, right=0.89)

    fig.savefig("./mlp_2_contour_and_node_trail.pdf")

    plt.close(fig)
    rcParams.update(rcParamsDefault)


def load_net(data_folder):
    net_params = jnp.load(data_folder + "/net_params.npy", allow_pickle=True)
    net_init, net_apply = get_net_arch_large_dim()
    return net_params, net_apply


if __name__ == "__main__":
    low, high = -2, 2

    data_folder = "../../datasets/MLP_2"
    conf = read_config_file(data_folder)
    net_params, net_apply = load_net(data_folder)
    data_generator = DataGeneratorMlp(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name="data_file.txt", test_folder=data_folder, d=conf["d"], sliding_window_size=conf["sliding_window_size"])
    # Figure 4 (b)
    draw_f_approx_contour_and_node_trail(net_apply, net_params, data_generator.data, low, high)
    draw_f_and_f_approx(net_apply, net_params, low, high)
