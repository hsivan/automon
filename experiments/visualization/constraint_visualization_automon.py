import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from automon.automon.automon_coordinator import AdcdHelper
from automon import AutomonNode
from test_utils.functions_to_monitor import func_entropy, func_variance, func_inner_product, func_rozenbrock
from test_utils.node_stream import NodeStreamFrequency, NodeStreamFirstAndSecondMomentum, NodeStreamAverage


def entropy_automon_draw_constraints(node):

    def prep_domain_grid():
        X_domain = np.arange(node.domain[0][0], node.domain[0][1], 0.02)
        Y_domain = np.arange(node.domain[0][0], node.domain[0][1], 0.02)
        X, Y = np.meshgrid(X_domain, Y_domain)
        mask_keep = X + Y <= 1
        residual_keep = 1 - X - Y
        residual_keep = np.clip(residual_keep, 0, 1)
        domain_grid_as_vector = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), residual_keep.reshape(-1, 1)), axis=1)
        return X, Y, domain_grid_as_vector, mask_keep

    def plot_points_on_graph(ax):
        ax.plot([node.x0[0]], [node.x0[1]], [func_entropy(node.x0)], marker='o', markersize=5, label='$p_0$', color="red")
        ax.plot([node.x[0]], [node.x[1]], [func_entropy(node.x)], marker='o', markersize=5, label='$p$ local', color="black")
        ax.plot([node.x0_local[0]], [node.x0_local[1]], [func_entropy(node.x0_local)], marker='o', markersize=5, label='$p_0$ local', color="green")

    def add_legend(ax):
        # Workaround to enable legend (no legend support for 3d surface)
        handles, labels = ax.get_legend_handles_labels()

        fake2Dline1 = Line2D([0], [0], linestyle="-", c='red')
        fake2Dline2 = Line2D([0], [0], linestyle="-", c='blue')
        fake2Dline3 = Line2D([0], [0], linestyle="-", c='yellow')
        fake2Dpoint1 = Line2D([0], [0], linestyle="none", c='red', marker='o')
        fake2Dpoint2 = Line2D([0], [0], linestyle="none", c='black', marker='o')
        fake2Dpoint3 = Line2D([0], [0], linestyle="none", c='green', marker='o')
        fake2Dpoint4 = Line2D([0], [0], linestyle="none", c='blue', marker='o')

        if len(labels) == 6:
            # Should not include the projection point (blue, or fake2Dpoint4)
            ax.legend([fake2Dpoint1, fake2Dpoint2, fake2Dpoint3, fake2Dline1, fake2Dline2, fake2Dline3], labels, numpoints=1)
        else:
            ax.legend([fake2Dpoint1, fake2Dpoint2, fake2Dpoint3, fake2Dpoint4, fake2Dline1, fake2Dline2, fake2Dline3], labels, numpoints=1)

    def draw_constraints_bounds():
        # Possible only if d=3
        assert (node.x.shape[0] == 3)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.suptitle('Node ' + str(node.idx) + ', $f(p)=-p_1 log(p_1)-p_2 log(p_2)-(1-p_1-p_2) log(1-p_1-p_2)$')
        ax.set(xlabel='$p_1$', ylabel=r'$p_2$', zlabel='Entropy')

        # Prepare domain grid for graph
        X, Y, domain_grid_as_vector, mask_keep = prep_domain_grid()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Plot the surface of the monitored function in the domain
            Z = func_entropy(domain_grid_as_vector)
            Z = Z.reshape(X.shape)
            Z[np.logical_not(mask_keep)] = np.nan
            ax.plot_surface(X, Y, Z, color="red", linewidth=0, antialiased=False, label='f(p)=$g(p)-h(p)$', alpha=0.2)

            # Plot the upper threshold
            ax.plot_surface(X, Y, np.ones_like(X) * node.u_thresh, color="blue", linewidth=0, antialiased=False, label='$U_{thresh}$', alpha=0.2)
            # Plot the lower threshold
            ax.plot_surface(X, Y, np.ones_like(X) * node.l_thresh, color="yellow", linewidth=0, antialiased=False, label='$L_{thresh}$', alpha=0.2)

            # Plot x0, x0_local and p
            plot_points_on_graph(ax)

        add_legend(ax)
        plt.show()

    def draw_upper_threshold_constraints():
        violation = "Violation"
        x = node._get_point_to_check()
        if node._inside_domain(x) and node._below_safe_zone_upper_bound(x):
            violation = "NO Violation"
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.suptitle('Node ' + str(node.idx) + ', Upper Threshold Constraint (' + violation + ')')
        ax.set(xlabel='$p_1$', ylabel=r'$p_2$', zlabel='Entropy')

        # Prepare domain grid for graph
        X, Y, domain_grid_as_vector, mask_keep = prep_domain_grid()

        # The hyperplane equation of the tangent plane to h at the point x0, plus the upper threshold is:
        # z(x) = h(x0) + grad_h(x0)*(x-x0) + u_thresh.
        # The OK area is:
        # g(x) <= z(x).
        # If true - f(x) is below the upper threshold (inside the safe zone).
        # Otherwise, f(x) is above the upper threshold (outside the safe zone).
        z_p = node.h_func(np.expand_dims(node.x0, axis=0)) + (domain_grid_as_vector - np.expand_dims(node.x0, axis=0)) @ np.expand_dims(node.h_func_grad_at_x0, axis=1) + node.u_thresh
        z_p = z_p.reshape(X.shape)
        z_p[np.logical_not(mask_keep)] = np.nan
        g_p = node.g_func(domain_grid_as_vector).reshape(X.shape)
        g_p[np.logical_not(mask_keep)] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Plot the tangent hyperplane
            ax.plot_surface(X, Y, z_p, color="red", linewidth=0, antialiased=False, label='Tangent $h(p)+U_{thresh}$', alpha=0.2)
            # Plot the lower g
            ax.plot_surface(X, Y, g_p, color="blue", linewidth=0, antialiased=False, label='$g(p)$', alpha=0.2)

            # Plot the h plus upper threshold
            h_p_plus_upper_thresh = node.h_func(domain_grid_as_vector).reshape(X.shape) + node.u_thresh
            h_p_plus_upper_thresh[np.logical_not(mask_keep)] = np.nan
            ax.plot_surface(X, Y, h_p_plus_upper_thresh, color="yellow", linewidth=0, antialiased=False, label='$h(p)+U_{thresh}$', alpha=0.2)

            # Plot x0, x0_local and x. Then plot projection of x0 on the tangent plane
            plot_points_on_graph(ax)
            ax.plot([node.x0[0]], [node.x0[1]], [node.h_func(node.x0) + node.u_thresh], marker='o', markersize=5, label='$h(p_0)+U_{thresh}$', color="blue")

        add_legend(ax)
        plt.show()

    def draw_lower_threshold_constraints():
        violation = "Violation"
        x = node._get_point_to_check()
        if node._inside_domain(x) and node._below_safe_zone_upper_bound(x):
            violation = "NO Violation"
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.suptitle('Node ' + str(node.idx) + ', Lower Threshold Constraint (' + violation + ')')
        ax.set(xlabel='$p_1$', ylabel=r'$p_2$', zlabel='Entropy')

        # Prepare domain grid for graph
        X, Y, domain_grid_as_vector, mask_keep = prep_domain_grid()

        # The hyperplane equation of the tangent plane to g at the point x0, minus the lower threshold is:
        # z(x) = g(x0) + grad_g(x0)*(x-x0) - l_thresh.
        # The OK area is:
        # h(x) <= z(x).
        # If true - f(x) is above the lower threshold (inside the safe zone).
        # Otherwise, f(x) is below the lower threshold (outside the safe zone).
        z_p = node.g_func(np.expand_dims(node.x0, axis=0)) + (domain_grid_as_vector - np.expand_dims(node.x0, axis=0)) @ np.expand_dims(node.g_func_grad_at_x0, axis=1) - node.l_thresh
        z_p = z_p.reshape(X.shape)
        z_p[np.logical_not(mask_keep)] = np.nan
        h_p = node.h_func(domain_grid_as_vector).reshape(X.shape)
        h_p[np.logical_not(mask_keep)] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Plot the tangent hyperplane
            ax.plot_surface(X, Y, z_p, color="red", linewidth=0, antialiased=False, label='Tangent $g(p)-L_{thresh}$', alpha=0.2)
            # Plot the lower h
            ax.plot_surface(X, Y, h_p, color="blue", linewidth=0, antialiased=False, label='$h(p)$', alpha=0.2)

            # Plot the g minus lower threshold
            g_p_minus_lower_thresh = node.g_func(domain_grid_as_vector).reshape(X.shape) - node.l_thresh
            g_p_minus_lower_thresh[np.logical_not(mask_keep)] = np.nan
            ax.plot_surface(X, Y, g_p_minus_lower_thresh, color="yellow", linewidth=0, antialiased=False, label='$g(p)-L_{thresh}$', alpha=0.2)

            # Plot x0, x0_local and x. Then plot projection of x0 on the tangent plane
            plot_points_on_graph(ax)
            ax.plot([node.x0[0]], [node.x0[1]], [node.g_func(node.x0) - node.l_thresh], marker='o', markersize=5, label="$g(p_0)-L_{thresh}$", color="blue")

        add_legend(ax)
        plt.show()

    if node.x.shape[0] == 3:
        draw_constraints_bounds()
        x = node._get_point_to_check()
        if not node._inside_domain(x) or not node._below_safe_zone_upper_bound(x):
            draw_upper_threshold_constraints()
        if not node._inside_domain(x) or not node._above_safe_zone_lower_bound(x):
            draw_lower_threshold_constraints()
    else:
        pass


def automon_draw_constraints(node, func_to_monitor, xlabel='$x$', ylabel='$y$', zlabel='f'):

    def prep_domain_grid():
        X_domain = np.arange(node.x0[0] - 3, node.x0[0] + 3, 0.2)
        Y_domain = np.arange(node.x0[1] - 3, node.x0[1] + 3, 0.2)
        X, Y = np.meshgrid(X_domain, Y_domain)
        domain_grid_as_vector = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
        return X, Y, domain_grid_as_vector

    def plot_points_on_graph(ax):
        ax.plot([node.x0[0]], [node.x0[1]], [func_to_monitor(node.x0)], marker='o', markersize=5, label='$x_0$', color="red")
        ax.plot([node.x[0]], [node.x[1]], [func_to_monitor(node.x)], marker='o', markersize=5, label='$x$ local', color="black")
        ax.plot([node.x0_local[0]], [node.x0_local[1]], [func_to_monitor(node.x0_local)], marker='o', markersize=5, label='$x0$ local', color="green")

    def add_legend(ax):
        # Workaround to enable legend (no legend support for 3d surface)
        handles, labels = ax.get_legend_handles_labels()

        fake2Dline1 = Line2D([0], [0], linestyle="-", c='red')
        fake2Dline2 = Line2D([0], [0], linestyle="-", c='blue')
        fake2Dline3 = Line2D([0], [0], linestyle="-", c='yellow')
        fake2Dpoint1 = Line2D([0], [0], linestyle="none", c='red', marker='o')
        fake2Dpoint2 = Line2D([0], [0], linestyle="none", c='black', marker='o')
        fake2Dpoint3 = Line2D([0], [0], linestyle="none", c='green', marker='o')
        fake2Dpoint4 = Line2D([0], [0], linestyle="none", c='blue', marker='o')

        if len(labels) == 6:
            # Should not include the projection point (blue, or fake2Dpoint4)
            ax.legend([fake2Dpoint1, fake2Dpoint2, fake2Dpoint3, fake2Dline1, fake2Dline2, fake2Dline3], labels, numpoints=1)
        else:
            ax.legend([fake2Dpoint1, fake2Dpoint2, fake2Dpoint3, fake2Dpoint4, fake2Dline1, fake2Dline2, fake2Dline3], labels, numpoints=1)

    def draw_constraints_bounds():
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.suptitle('Node ' + str(node.idx) + ', $f(x)$ and the Bounds')
        ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

        # Prepare domain grid for graph
        X, Y, domain_grid_as_vector = prep_domain_grid()

        # Plot the surface of the monitored function in the domain
        Z = func_to_monitor(domain_grid_as_vector)
        Z = Z.reshape(X.shape)
        ax.plot_surface(X, Y, Z, color="red", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)

        # Plot the upper threshold
        ax.plot_surface(X, Y, np.ones_like(X) * node.u_thresh, color="blue", linewidth=0, antialiased=False, label='$U_{thresh}$', alpha=0.2)
        # Plot the lower threshold
        ax.plot_surface(X, Y, np.ones_like(X) * node.l_thresh, color="yellow", linewidth=0, antialiased=False, label='$L_{thresh}$', alpha=0.2)

        # Plot x0, x0_local and x
        plot_points_on_graph(ax)

        add_legend(ax)
        plt.show()

    def draw_upper_threshold_constraints():
        violation = "Violation"
        x = node._get_point_to_check()
        if node._below_safe_zone_upper_bound(x):
            violation = "NO Violation"
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.suptitle('Node ' + str(node.idx) + ', Upper Threshold Constraint (' + violation + ')')
        ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

        # Prepare domain grid for graph
        X, Y, domain_grid_as_vector = prep_domain_grid()

        # The hyperplane equation of the tangent plane to h at the point x0, plus the upper threshold is:
        # z(x) = h(x0) + grad_h(x0)*(x-x0) + u_thresh.
        # The OK area is:
        # g(x) <= z(x).
        # If true - f(x) is below the upper threshold (inside the safe zone).
        # Otherwise, f(x) is above the upper threshold (outside the safe zone).
        z_x = node.h_func(np.expand_dims(node.x0, axis=0)) + (domain_grid_as_vector - np.expand_dims(node.x0, axis=0)) @ np.expand_dims(node.h_func_grad_at_x0, axis=1) + node.u_thresh
        z_x = z_x.reshape(X.shape)
        g_x = node.g_func(domain_grid_as_vector).reshape(X.shape)

        # Plot the tangent hyperplane
        ax.plot_surface(X, Y, z_x, color="red", linewidth=0, antialiased=False, label='Tangent $h(x)+U_{thresh}$', alpha=0.2)
        # Plot the lower g
        ax.plot_surface(X, Y, g_x, color="blue", linewidth=0, antialiased=False, label='$g(x)$', alpha=0.2)

        # Plot the h plus upper threshold
        h_x_plus_upper_thresh = node.h_func(domain_grid_as_vector).reshape(X.shape) + node.u_thresh
        ax.plot_surface(X, Y, h_x_plus_upper_thresh, color="yellow", linewidth=0, antialiased=False, label='$h(x)+U_{thresh}$', alpha=0.2)

        # Plot x0, x0_local and x. Then plot projection of x0 on the tangent plane
        plot_points_on_graph(ax)
        ax.plot([node.x0[0]], [node.x0[1]], [node.h_func(node.x0) + node.u_thresh], marker='o', markersize=5, label='$h(x_0)+U_{thresh}$', color="blue")

        add_legend(ax)
        plt.show()

    def draw_lower_threshold_constraints():
        violation = "Violation"
        x = node._get_point_to_check()
        if node._above_safe_zone_lower_bound(x):
            violation = "NO Violation"
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.suptitle('Node ' + str(node.idx) + ', Lower Threshold Constraint (' + violation + ')')
        ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

        # Prepare domain grid for graph
        X, Y, domain_grid_as_vector = prep_domain_grid()

        # The hyperplane equation of the tangent plane to g at the point x0, minus the lower threshold is:
        # z(x) = g(x0) + grad_g(x0)*(x-x0) - l_thresh.
        # The OK area is:
        # h(x) <= z(x).
        # If true - f(x) is above the lower threshold (inside the safe zone).
        # Otherwise, f(x) is below the lower threshold (outside the safe zone).
        z_x = node.g_func(np.expand_dims(node.x0, axis=0)) + (domain_grid_as_vector - np.expand_dims(node.x0, axis=0)) @ np.expand_dims(node.g_func_grad_at_x0, axis=1) - node.l_thresh
        z_x = z_x.reshape(X.shape)
        h_x = node.h_func(domain_grid_as_vector).reshape(X.shape)

        # Plot the tangent hyperplane
        ax.plot_surface(X, Y, z_x, color="red", linewidth=0, antialiased=False, label='Tangent $g(x)-L_{thresh}$',
                        alpha=0.2)
        # Plot the lower h
        ax.plot_surface(X, Y, h_x, color="blue", linewidth=0, antialiased=False, label='$h(x)$', alpha=0.2)

        # Plot the g minus lower threshold
        g_x_minus_lower_thresh = node.g_func(domain_grid_as_vector).reshape(X.shape) - node.l_thresh
        ax.plot_surface(X, Y, g_x_minus_lower_thresh, color="yellow", linewidth=0, antialiased=False, label='$g(x)-L_{thresh}$', alpha=0.2)

        # Plot x0, x0_local and x. Then plot projection of x0 on the tangent plane
        plot_points_on_graph(ax)
        ax.plot([node.x0[0]], [node.x0[1]], [node.g_func(node.x0) - node.l_thresh], marker='o', markersize=5, label="$g(x_0)-L_{thresh}$", color="blue")

        add_legend(ax)
        plt.show()

    if node.x.shape[0] == 2:  # local vector has only two attributes x1 and x2
        draw_constraints_bounds()
        '''x = node._get_point_to_check()
        if not node._below_safe_zone_upper_bound(x):
            draw_upper_threshold_constraints()
        if not node._above_safe_zone_lower_bound(x):
            draw_lower_threshold_constraints()'''
        draw_upper_threshold_constraints()
        draw_lower_threshold_constraints()


def visualize_entropy():
    adcd_helper = AdcdHelper(func_entropy)

    epsilon = 0.0000001

    # 3 dimensional case
    k = 3
    sliding_window_size = 5
    node_idx = 1
    slack = np.zeros(k)  # No slack
    x0 = np.array([1, 1, 4]) / np.sum([1, 1, 4])
    domain = [(0+epsilon, 1-epsilon)] * k
    dc_type, extreme_lambda = adcd_helper.adcd_x(x0, domain, 0)

    node3 = AutomonNode(idx=node_idx, d=k, max_f_val=func_entropy(np.ones(k, dtype=np.float) / k), min_f_val=0.0, domain=(0 + epsilon, 1 - epsilon), func_to_monitor=func_entropy)
    node3.sync(x0, slack, 0.7, 1, -1, dc_type, extreme_lambda)
    entropy_automon_draw_constraints(node3)
    # Fill sliding window
    node_stream = NodeStreamFrequency(2, sliding_window_size, 1, x0.shape[0])
    for i in range(sliding_window_size):
        node_stream.set_new_data_point(0, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node3.set_new_data_point(local_vector)
    entropy_automon_draw_constraints(node3)
    assert(not b_inside_safe_zone)
    node_stream.set_new_data_point(1, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node3.set_new_data_point(local_vector)
    entropy_automon_draw_constraints(node3)
    assert(not b_inside_safe_zone)
    node_stream.set_new_data_point(2, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node3.set_new_data_point(local_vector)
    entropy_automon_draw_constraints(node3)
    assert(not b_inside_safe_zone)
    node_stream.set_new_data_point(2, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node3.set_new_data_point(local_vector)
    entropy_automon_draw_constraints(node3)
    assert(not b_inside_safe_zone)
    node_stream.set_new_data_point(2, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node3.set_new_data_point(local_vector)
    entropy_automon_draw_constraints(node3)
    assert b_inside_safe_zone
    node3.sync(x0, slack, 0.7, 1, -1, dc_type, extreme_lambda)
    entropy_automon_draw_constraints(node3)

    # 2 dimensional case
    k = 2
    sliding_window_size = 4
    slack = np.zeros(k)  # No slack
    node_idx = 1
    x0 = np.array([1, 4]) / np.sum([1, 4])
    domain = [(0 + epsilon, 1 - epsilon)] * k
    dc_type, extreme_lambda = adcd_helper.adcd_x(x0, domain, 0)

    node2 = AutomonNode(idx=node_idx, d=k, max_f_val=func_entropy(np.ones(k, dtype=np.float) / k), min_f_val=0.0, func_to_monitor=func_entropy)
    node2.sync(x0, slack, 0.5, 0.6, -1, dc_type, extreme_lambda)
    entropy_automon_draw_constraints(node2)
    # Fill sliding window
    node_stream = NodeStreamFrequency(2, sliding_window_size, 1, x0.shape[0])
    for i in range(sliding_window_size):
        node_stream.set_new_data_point(0, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node2.set_new_data_point(local_vector)
    entropy_automon_draw_constraints(node2)
    assert(not b_inside_safe_zone)
    node_stream.set_new_data_point(1, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node2.set_new_data_point(local_vector)
    entropy_automon_draw_constraints(node2)
    assert(not b_inside_safe_zone)
    node_stream.set_new_data_point(1, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node2.set_new_data_point(local_vector)
    entropy_automon_draw_constraints(node2)
    assert(not b_inside_safe_zone)
    node_stream.set_new_data_point(1, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node2.set_new_data_point(local_vector)
    entropy_automon_draw_constraints(node2)
    assert b_inside_safe_zone
    node_stream.set_new_data_point(1, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node2.set_new_data_point(local_vector)
    entropy_automon_draw_constraints(node2)
    assert not b_inside_safe_zone
    node2.sync(x0, slack, 0.5, 0.6, -1, dc_type, extreme_lambda)
    entropy_automon_draw_constraints(node2)


def visualize_variance():
    adcd_helper = AdcdHelper(func_variance)

    slack = np.zeros(2)  # No slack
    x0 = np.array([0.5, 2])
    sliding_window_size = 5
    node_idx = 1

    dc_type, extreme_lambda = adcd_helper.adcd_x(x0, None, 0)
    node = AutomonNode(idx=node_idx, d=2, func_to_monitor=func_variance, min_f_val=0.0)
    node.sync(x0, slack, 0.08, 3, -1, dc_type, extreme_lambda)
    # Fill sliding window
    node_stream = NodeStreamFirstAndSecondMomentum(2, sliding_window_size, 1, x0.shape[0])
    for i in range(sliding_window_size):
        node_stream.set_new_data_point(0, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node.set_new_data_point(local_vector)
    automon_draw_constraints(node, func_variance, xlabel='E$[x]$', ylabel=r'E$[x^2]$', zlabel='Var')
    assert (not b_inside_safe_zone)
    node_stream.set_new_data_point(2, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node.set_new_data_point(local_vector)
    automon_draw_constraints(node, func_variance, xlabel='E$[x]$', ylabel=r'E$[x^2]$', zlabel='Var')
    assert b_inside_safe_zone
    node_stream.set_new_data_point(6, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node.set_new_data_point(local_vector)
    automon_draw_constraints(node, func_variance, xlabel='E$[x]$', ylabel=r'E$[x^2]$', zlabel='Var')
    assert (not b_inside_safe_zone)


def visualize_rozenbrock():
    adcd_helper = AdcdHelper(func_rozenbrock)

    slack = np.zeros(2)  # No slack
    x0 = np.array([0.5, 2])
    sliding_window_size = 5
    node_idx = 1

    dc_type, extreme_lambda = adcd_helper.adcd_x(x0, None, 0)
    node = AutomonNode(idx=node_idx, d=x0.shape[0], func_to_monitor=func_rozenbrock)
    node.sync(x0, slack, 0.08, 3, -1, dc_type, extreme_lambda)
    # Fill sliding window
    node_stream = NodeStreamAverage(2, sliding_window_size, x0.shape[0], x0.shape[0])
    for i in range(sliding_window_size):
        node_stream.set_new_data_point(0, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node.set_new_data_point(local_vector)
    automon_draw_constraints(node, func_rozenbrock)
    assert (not b_inside_safe_zone)


def visualize_inner_product():
    adcd_helper = AdcdHelper(func_inner_product)

    slack = np.zeros(2)  # No slack
    x0 = np.array([0.5, 2])
    sliding_window_size = 5
    node_idx = 1

    dc_type, extreme_lambda = adcd_helper.adcd_x(x0, None, 0)
    node = AutomonNode(idx=node_idx, d=x0.shape[0], func_to_monitor=func_inner_product)
    node.sync(x0, slack, 0.08, 3, -1, dc_type, extreme_lambda)
    # Fill sliding window
    node_stream = NodeStreamAverage(2, sliding_window_size, x0.shape[0], x0.shape[0])
    for i in range(sliding_window_size):
        node_stream.set_new_data_point(np.array([0, 0]), node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node.set_new_data_point(local_vector)
    automon_draw_constraints(node, func_inner_product)
    assert (not b_inside_safe_zone)


if __name__ == "__main__":
    plt.close("all")
    visualize_entropy()
    visualize_variance()
    visualize_rozenbrock()
    visualize_inner_product()
