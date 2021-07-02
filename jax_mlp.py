import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Tanh
from jax.experimental import optimizers
from jax import jit, grad
from jax import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm


def func_to_approx(x):
    y = x[:, 0] * jnp.exp(-jnp.sum(x**2, axis=1) / (x.shape[1] - 1))
    return y


def plot_surface(func_name, x1_grid, x2_grid, f, low, high):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([low, high])
    ax.set_ylim([low, high])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel(func_name)
    ax.plot_surface(x1_grid, x2_grid, f, cmap=cm.coolwarm)
    return fig


def draw_f_and_f_approx(net_apply, network_params, low, high, test_folder=None):
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

    fig_f = plot_surface('$f=x_1 exp(-x_1^2 - x_2^2)$', x1_grid, x2_grid, f, low, high)
    fig_f_approx = plot_surface(r'trained $\tilde{f}$', x1_grid, x2_grid, f_predictions, low, high)

    if test_folder is not None:
        fig_f_approx.savefig(test_folder + "/approx_func_surface.pdf")
    else:
        plt.show()

    plt.close(fig_f)
    plt.close(fig_f_approx)


def draw_f_approx_contour_and_node_trail(net_apply, network_params, nodes_data, test_folder=None, low=-2, high=2):
    node1_trail = nodes_data[::10]  # trail of first node
    node6_trail = nodes_data[5::10]  # trail of sixth node

    # Average 20 first and last samples
    node1_start = jnp.mean(node1_trail[:20], axis=0)
    node1_end = jnp.mean(node1_trail[-20:], axis=0)
    node6_start = jnp.mean(node6_trail[:20], axis=0)
    node6_end = jnp.mean(node6_trail[-20:], axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    x1 = jnp.linspace(low, high, 50)
    x2 = jnp.linspace(low, high, 50)
    x1_grid, x2_grid = jnp.meshgrid(x1, x2)
    x_ = jnp.append(x1_grid.reshape(x1_grid.shape[0] ** 2, -1), x2_grid.reshape(x2_grid.shape[0] ** 2, -1), axis=1)
    f_predictions = net_apply(network_params, x_)
    f_predictions = f_predictions.reshape(x1_grid.shape[0], -1)
    plt.contourf(x1_grid, x2_grid, f_predictions, 20, cmap='RdGy')
    plt.colorbar()
    plt.scatter(node1_trail[:, 0], node1_trail[:, 1], s=0.5)
    plt.scatter(node6_trail[:, 0], node6_trail[:, 1], s=0.5)
    plt.arrow(node1_start[0], node1_start[1], node1_end[0]-node1_start[0], node1_end[1]-node1_start[1], color='navy', head_width=0.1, head_length=0.15)
    plt.arrow(node6_start[0], node6_start[1], node6_end[0] - node6_start[0], node6_end[1] - node6_start[1], color='brown', head_width=0.1, head_length=0.15)

    blue_line = mlines.Line2D([], [], color='navy', label='node1 trail')
    orange_line = mlines.Line2D([], [], color='brown', label='node6 trail')
    plt.legend(handles=[blue_line, orange_line])

    if test_folder is not None:
        fig.savefig(test_folder + "/approx_func_contour_and_node_trail.pdf")
    else:
        plt.show()


def get_net_arch_large_dim():
    net_init, net_apply = stax.serial(
        Dense(40), Tanh,
        Dense(15), Tanh,
        Dense(5), Tanh,
        Dense(1)
    )
    return net_init, net_apply


def train_net(test_folder=None, x_dim=2, num_train_iter=5000, step_size=1e-2):
    # Use stax to set up network initialization and evaluation functions
    net_init, net_apply = get_net_arch_large_dim()

    # MSE loss
    def loss_squared_error(params, batch):
        inputs_, targets_ = batch
        predictions = net_apply(params, inputs_)
        return jnp.mean((predictions.squeeze() - targets_) ** 2)

    # Define a compiled update step
    @jit
    def step(iteration, opt_state_, batch):
        params = get_params(opt_state_)
        g = grad(loss_squared_error)(params, batch)
        return opt_update(iteration, g, opt_state_)

    @jit
    def next_training_batch(rnd):
        x = random.normal(rnd, shape=(500, x_dim))
        y = func_to_approx(x)
        return x, y

    # Initialize parameters, not committing to a batch shape
    rnd = random.PRNGKey(0)
    in_shape = (-1, x_dim)
    out_shape, net_params = net_init(rnd, in_shape)

    # Use optimizers to set optimizer initialization and update functions
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    # Optimize parameters in a loop
    opt_state = opt_init(net_params)
    for i in range(num_train_iter):
        training_batch = next_training_batch(rnd)
        opt_state = step(i, opt_state, training_batch)
        if i % 500 == 0:
            net_params = get_params(opt_state)
            loss = loss_squared_error(net_params, training_batch)
            print(i, loss)
    net_params = get_params(opt_state)

    print(net_params)
    if x_dim == 2:
        draw_f_and_f_approx(net_apply, net_params, -2, 2, test_folder)

    save_net(test_folder, net_params)

    return net_params, net_apply


def load_net(test_folder):
    net_params = jnp.load(test_folder + "/net_params.npy", allow_pickle=True)
    net_init, net_apply = get_net_arch_large_dim()

    return net_params, net_apply


def save_net(test_folder, net_params):
    if test_folder is not None:
        jnp.save(test_folder + "/net_params", net_params)

if __name__ == "__main__":
    neteork_params, net_apply_fun = train_net(test_folder="./", x_dim=20, num_train_iter=100000, step_size=1e-4)
    load_net(test_folder="./")
