import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from matplotlib.lines import Line2D
import numpy as np
from automon import AutomonNode
from test_utils.functions_to_monitor import func_inner_product
from experiments.visualization.visualization_utils import get_figsize
from automon.automon.automon_coordinator import AdcdHelper


def prep_domain_grid():
    X_domain = np.arange(-1, 1.02, 0.1)
    Y_domain = np.arange(-1, 1.02, 0.1)
    X, Y = np.meshgrid(X_domain, Y_domain)
    domain_grid_as_vector = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    return X, Y, domain_grid_as_vector


def plot_points_on_graph(node, ax):
    ax.plot([node.x0[0]], [node.x0[1]], [func_inner_product(node.x0)], marker='o', markersize=3, label='$f(x_0,y_0)$', color="red")
    ax.plot([node.x[0]], [node.x[1]], [func_inner_product(node.x)], marker='o', markersize=3, label='$f(x,y)$', color="black")


def add_legend(ax):
    # Workaround to enable legend (no legend support for 3d surface)
    handles, labels = ax.get_legend_handles_labels()

    fake2Dline1 = Line2D([0], [0], linestyle="-", c='red')
    fake2Dline2 = Line2D([0], [0], linestyle="-", c='blue')
    fake2Dline3 = Line2D([0], [0], linestyle="-", c='yellow')
    fake2Dline4 = Line2D([0], [0], linestyle="-", c='green')
    fake2Dpoint1 = Line2D([0], [0], linestyle="none", c='red', marker='o', markersize=3)
    fake2Dpoint2 = Line2D([0], [0], linestyle="none", c='black', marker='o', markersize=3)

    if len(labels) == 5:
        # Should not include the projection point (blue, or fake2Dpoint4)
        ax.legend([fake2Dpoint1, fake2Dpoint2, fake2Dline1, fake2Dline2, fake2Dline3], labels, numpoints=1, loc="center left")
    else:
        ax.legend([fake2Dpoint1, fake2Dpoint2, fake2Dline2, fake2Dline4], labels, numpoints=1, loc="center left")


def draw_constraints_bounds(node):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 6})
    rcParams.update({'font.size': 7})

    fig = plt.figure(figsize=get_figsize(wf=0.7))
    ax = fig.add_subplot(projection='3d')
    ax.set(xlabel='$x$', ylabel='$y$')
    ax.azim = -35  # default -60
    ax.elev = 3  # default 30
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zlim([-0.8, 1.2])

    # Prepare domain grid for graph
    X, Y, domain_grid_as_vector = prep_domain_grid()

    # Plot the surface of the monitored function in the domain
    Z = func_inner_product(domain_grid_as_vector)
    Z = Z.reshape(X.shape)
    ax.plot_surface(X, Y, Z, color="red", linewidth=0, antialiased=False, label='$f$', alpha=0.2)

    # Plot the upper threshold
    ax.plot_surface(X, Y, np.ones_like(X) * node.u_thresh, color="blue", linewidth=0, antialiased=False, label='$U$', alpha=0.2)
    # Plot the lower threshold
    ax.plot_surface(X, Y, np.ones_like(X) * node.l_thresh, color="yellow", linewidth=0, antialiased=False, label='$L$', alpha=0.2)

    # Plot x0 and x
    plot_points_on_graph(node, ax)

    add_legend(ax)
    plt.subplots_adjust(top=1.2, bottom=0.07, left=0.01, right=0.9)
    fig.savefig("inner_product_func_and_bounds.pdf")

    rcParams.update(rcParamsDefault)


def draw_upper_threshold_constraints(node):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 6})
    rcParams.update({'font.size': 7})

    violation = "Violation"
    x = node._get_point_to_check()
    if node._below_safe_zone_upper_bound(x):
        violation = "no_Violation"
    fig = plt.figure(figsize=get_figsize(wf=0.7))
    ax = fig.add_subplot(projection='3d')
    ax.set(xlabel='$x$', ylabel='$y$')
    ax.azim = -35  # default -60
    ax.elev = 3  # default 30
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zlim([-0.8, 1.2])

    # Prepare domain grid for graph
    X, Y, domain_grid_as_vector = prep_domain_grid()

    # The hyperplane equation of the tangent plane to h at the point x0, plus the upper threshold is:
    # z(x) = h(x0) + grad_h(x0)*(x-x0) + u_thresh.
    # The OK area is:
    # g(x) <= z(x).
    # If true - f(x) is below the upper threshold (inside the safe zone).
    # Otherwise, f(x) is above the upper threshold (outside the safe zone).
    g_x = node.g_func(domain_grid_as_vector).reshape(X.shape)

    # Plot the upper threshold
    ax.plot_surface(X, Y, np.ones_like(X) * node.u_thresh, color="blue", linewidth=0, antialiased=False, label='$U$', alpha=0.2)
    # Plot the lower g
    ax.plot_surface(X, Y, g_x, color="green", linewidth=0, antialiased=False, label=r"$\breve{g}(x)$", alpha=0.2)

    # Plot x0 and x
    plot_points_on_graph(node, ax)

    add_legend(ax)
    plt.subplots_adjust(top=1.2, bottom=0.07, left=0.01, right=0.9)
    fig.savefig("inner_product_upper_constraint_" + violation + ".pdf")

    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    plt.close("all")
    adcd_helper = AdcdHelper(func_inner_product)

    slack = np.zeros(2)  # No slack
    x0 = np.array([0.1, 0.1])
    x = np.array([0.65, 0.65])

    dc_type, signed_H = adcd_helper.adcd_e(x0)
    node = AutomonNode(idx=1, x0_len=x0.shape[0], func_to_monitor=func_inner_product)
    node.sync(x0, slack, func_inner_product(x0)-0.5, func_inner_product(x0)+0.5, -1, dc_type, signed_H)
    b_inside_safe_zone = node.set_new_data_point(x)
    # Inside safe zone
    draw_constraints_bounds(node)
    draw_upper_threshold_constraints(node)
    assert b_inside_safe_zone

    dc_type, extreme_lambda = adcd_helper.adcd_x(x0, None, 0)
    node = AutomonNode(idx=1, x0_len=x0.shape[0], func_to_monitor=func_inner_product)
    node.sync(x0, slack, func_inner_product(x0) - 0.5, func_inner_product(x0) + 0.5, -1, dc_type, extreme_lambda)
    b_inside_safe_zone = node.set_new_data_point(x)
    # Not inside safe zone
    draw_upper_threshold_constraints(node)
    assert not b_inside_safe_zone
