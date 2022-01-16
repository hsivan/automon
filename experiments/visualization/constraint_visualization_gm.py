import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from test_utils.functions_to_monitor import func_entropy, func_variance
from automon import GmEntropyNode, GmVarianceNode
from test_utils.node_stream import NodeStreamFrequency, NodeStreamFirstAndSecondMomentum


def variance_gm_draw_constraints(node):
    lower_x_lim = np.min((-3, - 1 * np.abs(node.x[0])))
    upper_x_lim = -1 * lower_x_lim
    fig, ax = plt.subplots()
    fig.suptitle('Node ' + str(node.idx))
    ax.set(xlabel='E$[x]$', ylabel=r'E$[x^2]$')
    ax.set_aspect('equal')
    ax.set_xlim([lower_x_lim, upper_x_lim])

    # Plot admissible region
    x = np.linspace(lower_x_lim, upper_x_lim, 500)
    y_l = node._calc_parabola(node.l_thresh, x)
    y_u = node._calc_parabola(node.u_thresh, x)
    ax.plot(x, y_l, label='lower bound')
    ax.plot(x, y_u, label='upper bound')
    ax.fill_between(x, y_l, y_u, alpha=0.3)

    # Plot the tangent line at the point q
    slop = 2 * node.q[0]
    tangent_y_intersection = node.q[1] - slop * node.q[0]
    tangent = slop * x + tangent_y_intersection
    ax.plot(x, tangent, label='tangent', color='green')

    # Plot the normal from x0 to q
    ax.plot([node.x0[0], node.q[0]], [node.x0[1], node.q[1]], 'r-')

    # Fill the safe zone with color
    ax.fill_between(x, y_l, tangent, where=tangent >= y_l, alpha=0.3)

    # Make sure the tangent and normal are orthogonal (the inner product is 0)
    normal = node.x0 - node.q  # The normal to the tangent line at q
    tangent_y_intersection = np.array([0, tangent_y_intersection])
    tangent = tangent_y_intersection - node.q  # The tangent line at q
    inner_prod = normal @ tangent
    assert(inner_prod < 1e-10)

    # Plot points x0, x, q
    ax.plot(node.x0[0], node.x0[1], 'ro', label='$v_0$')
    ax.plot(node.x0_local[0], node.x0_local[1], 'o', color='purple', markersize=5.5, label='$v_0$ local')
    ax.plot(node.x[0], node.x[1], 'bo', label='$v$ local', markersize=5)
    ax.plot(node.q[0], node.q[1], 'go', label='$q$', markersize=4.5)

    # Plot all roots (could be more than 1, which is q)
    ax.plot(node.roots, node._calc_parabola(node.u_thresh, node.roots), 'x', color='black', label='roots', markersize=4)

    fig.legend()
    fig.show()


def entropy_gm_draw_constraints(node):

    def entropy_gm_draw_constraints_3d():
        # Possible only if d=3
        assert (node.d == 3)

        epsilon = 0.00000000000001
        lower_x_lim = 0 + epsilon
        upper_x_lim = 1 - epsilon

        num_points_for_grid_axis = 500  # Increase this to get a better looking figure (to 1000)
        num_contour_levels = 50  # Increase this to get a better looking figure (to 100)
        p1 = np.linspace(lower_x_lim, upper_x_lim, num_points_for_grid_axis)
        p2 = np.linspace(lower_x_lim, upper_x_lim, num_points_for_grid_axis)
        p1_grid, p2_grid = np.meshgrid(p1, p2)
        mask_keep = p1_grid + p2_grid <= 1
        residual_keep = 1 - p1_grid - p2_grid
        residual_keep = np.clip(residual_keep, 0, 1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        fig.suptitle('Node ' + str(node.idx) + ' Entropy$=-p_1 log(p_1)-p_2 log(p_2)-(1-p_1-p_2) log(1-p_1-p_2)$')
        ax.set_xlabel('$p_1$')
        ax.set_ylabel('$p_2$')
        ax.set_zlabel('Entropy')
        p = np.concatenate(
            (p1_grid.reshape(num_points_for_grid_axis ** 2, 1), p2_grid.reshape(num_points_for_grid_axis ** 2, 1)),
            axis=1)
        p = np.concatenate((p, residual_keep.reshape(num_points_for_grid_axis ** 2, 1)), axis=1)
        f = func_entropy(p).reshape(num_points_for_grid_axis, num_points_for_grid_axis)
        f[np.logical_not(mask_keep)] = 0

        f_l = f.copy()
        f_l[f < node.l_thresh] = np.nan
        f_max = func_entropy(np.ones(node.d, dtype=float) / node.d)
        levels = np.linspace(node.l_thresh, f_max, num_contour_levels)
        ax.contour(p1, p2, f_l, alpha=0.05, colors='blue', levels=levels)
        projection = np.zeros_like(f_l)
        projection[f < node.l_thresh] = np.nan
        projection[np.logical_not(mask_keep)] = np.nan
        ax.contourf(p1, p2, projection, alpha=0.2, colors='blue', levels=1)

        f_u = f.copy()
        f_u[f > node.u_thresh] = np.nan
        f_u[f < epsilon] = np.nan
        levels = np.linspace(0, node.u_thresh, num_contour_levels)
        ax.contour(p1, p2, f_u, alpha=0.2, colors='orange', levels=levels)
        projection = np.zeros_like(f_l)
        projection[f > node.u_thresh] = np.nan
        projection[np.logical_not(mask_keep)] = np.nan
        ax.contourf(p1, p2, projection, alpha=0.2, colors='orange', levels=1)

        # Plot points x0, x, q
        p = node.x0
        f = func_entropy(p)
        ax.scatter(p[0], p[1], f, marker='o', s=50, label='$p_0$', color='red')
        ax.scatter(p[0], p[1], 0, marker='o', s=50, label='$p_0$ proj', color='red', alpha=0.6)

        p = node.q
        f = func_entropy(p)
        ax.scatter(p[0], p[1], f, marker='o', s=50, label='$q$', color='black')
        ax.scatter(p[0], p[1], 0, marker='o', s=50, label='$q$ proj', color='black', alpha=0.5)

        p = node.x0_local
        f = func_entropy(p)
        # ax.scatter(p[0], p[1], f, marker='o', s=20, label='$p_0$ local', color='green')
        ax.scatter(p[0], p[1], 0, marker='o', s=20, label='$p_0$ local proj', color='green')

        p = node.x
        f = func_entropy(p)
        # ax.scatter(p[0], p[1], f, marker='o', s=10, label='$p$ local', color='yellow', edgecolors='black', linewidths=0.3)
        ax.scatter(p[0], p[1], 0, marker='o', s=10, label='$p$ local proj', color='yellow', edgecolors='black',
                   linewidths=0.3)

        # Plot the normal from x0 projection to q projection
        ax.plot([node.x0[0], node.q[0]], [node.x0[1], node.q[1]], 'r-', alpha=0.6)

        # Draw the projection of the tangent hyperplane at the point q.
        # This means drawing p1,p2 where f(q) = derivative_p1(q)*p1 + derivative_p2(q)*p2 + derivative_p3(q)*p3 + b
        # and b = f(q) - derivative_p1(q)*q[0] - derivative_p2*q[1]
        fq = func_entropy(node.q)
        # The normal to the tangent hyperplane at the point q is the gradient of the function f (entropy) at the point q
        derivative = node.q.copy()
        derivative[node.q > 0] = -1.0 * np.log(node.q[node.q > 0]) - 1
        derivative_p1 = derivative[0]
        derivative_p2 = derivative[1]
        derivative_p3 = derivative[2]
        # The intersection of the tangent hyperplane with z axis
        b = fq - derivative_p1 * node.q[0] - derivative_p2 * node.q[1] - derivative_p3 * node.q[2]
        # Tangent line at q projected on xy plane

        if derivative_p3 - derivative_p2 != 0 and derivative_p3 - derivative_p1 != 0:
            p2_for_p1_0 = (derivative_p3 + b - fq) / (derivative_p3 - derivative_p2)
            p1_for_p2_0 = (derivative_p3 + b - fq) / (derivative_p3 - derivative_p1)
            if node.q[0] != 0:
                slop = (node.q[1] - p2_for_p1_0) / node.q[0]
            else:
                slop = node.q[1] / (node.q[0] - p1_for_p2_0)
            intersection = node.q[1] - slop * node.q[0]
            # Works if the tangent isn't parallel to x axis
            x = np.linspace(0, 1, 500)
            y = slop * x + intersection
            x = x[np.logical_and(y >= -0.1, y <= 1.1)]
            y = y[np.logical_and(y >= -0.1, y <= 1.1)]
            ax.plot(x, y, 'r-', alpha=0.6)
            # Works if the tangent isn't parallel to y axis
            y = np.linspace(0, 1, 500)
            x = (y - intersection) / slop
            x_ = x[np.logical_and(x >= -0.1, x <= 1.1)]
            y_ = y[np.logical_and(x >= -0.1, x <= 1.1)]
            ax.plot(x_, y_, 'r-', alpha=0.6)

        handles, labels = ax.get_legend_handles_labels()
        colors = ['blue', 'orange']
        new_handles = [Line2D([0], [0], color=c, alpha=0.4, linewidth=3, linestyle='solid') for c in colors]
        new_labels = ['lower bound + proj', 'upper bound + proj']
        handles += new_handles
        labels += new_labels
        fig.legend(handles, labels)

        fig.show()

    def entropy_gm_draw_constraints_2d():
        # Possible only if d=2
        assert (node.d == 2)

        lower_x_lim = 0
        upper_x_lim = 1
        fig, ax = plt.subplots()
        fig.suptitle('Node ' + str(node.idx))
        fig.suptitle('Node ' + str(node.idx) + ' Entropy$=-p_1 log(p_1)-(1-p_1) log(1-p_1)$')
        ax.set(xlabel='$p_1$', ylabel=r'Entropy')
        ax.set_aspect('equal')
        ax.set_xlim([lower_x_lim, upper_x_lim])

        # Plot admissible region
        p1 = np.expand_dims(np.linspace(lower_x_lim, upper_x_lim, 500), axis=0)
        p2 = 1 - p1
        p3 = np.concatenate((p1, p2), axis=0)
        p3 = p3.T
        f = func_entropy(p3)
        p = p3[:, 0]
        ax.plot(p, f, label='f', color='grey')
        ax.plot(p, np.ones(p.shape) * node.l_thresh, label='lower bound + proj', color='blue')
        ax.plot(p, np.ones(p.shape) * node.u_thresh, label='upper bound + proj', color='orange')
        ax.fill_between(p, np.ones(p.shape) * node.l_thresh, np.ones(p.shape) * node.u_thresh,
                        where=np.logical_and(f <= node.u_thresh, f >= node.l_thresh), alpha=0.3)

        # Plot points x0, x, q
        ax.plot(node.x0[0], func_entropy(node.x0), 'ro', label='$p_0$', markersize=7)
        ax.plot(node.x0[0], 0, 'ro', label='$x_0$ proj', alpha=0.6, markersize=7)

        ax.plot(node.x0_local[0], func_entropy(node.x0_local), 'o', color='green', markersize=5.5, label='$x_0$ local')

        ax.plot(node.x[0], func_entropy(node.x), 'o', label='$p$ local', markersize=4, color='yellow',
                markeredgecolor='black', markeredgewidth=0.5)

        ax.plot(node.q[0], func_entropy(node.q), 'o', label='$q$', markersize=4.5, color='black')
        ax.plot(node.q[0], 0, 'o', label='$q$ proj', markersize=4.5, alpha=0.6, color='black')

        # Draw projection of the constraints on x axis.
        # Upper bound is two parts: 0 to left_intersection_of_upper_bound_with_f and right_intersection_of_upper_bound_with_f to 1.
        # It is possible that the upper bound is one part if it is greater than the maximum of f.
        eps = 0.005
        if np.all(node.u_thresh <= f):
            # One part - all x axis
            x = np.linspace(0, 1, 100)
            ax.plot(x, np.zeros(x.shape), color='orange', alpha=0.5, linewidth=3)
        else:
            # Two parts
            x = np.linspace(0, p[np.logical_and(f + eps >= node.u_thresh, f - eps <= node.u_thresh)][0], 100)
            ax.plot(x, np.zeros(x.shape), color='orange', alpha=0.5, linewidth=3)
            x = np.linspace(p[np.logical_and(f + eps >= node.u_thresh, f - eps <= node.u_thresh)][-1], 1, 100)
            ax.plot(x, np.zeros(x.shape), color='orange', alpha=0.5, linewidth=3)
        # Lower bound is one part: left_intersection_of_upper_bound_with_f to right_intersection_of_upper_bound_with_f
        x = np.linspace(p[np.logical_and(f + eps >= node.l_thresh, f - eps <= node.l_thresh)][0],
                        p[np.logical_and(f + eps >= node.l_thresh, f - eps <= node.l_thresh)][-1], 100)
        ax.plot(x, np.zeros(x.shape), color='blue', alpha=0.3, linewidth=6)

        fig.legend()
        fig.show()

    if node.d == 2:
        entropy_gm_draw_constraints_2d()
    elif node.d == 3:
        entropy_gm_draw_constraints_3d()
    else:
        pass


def visualize_entropy():
    slack = np.zeros(3)  # No slack
    node_idx = 1

    # 3 dimensional case
    sliding_window_size = 5
    node3 = GmEntropyNode(idx=node_idx, d=3, func_to_monitor=func_entropy)
    x0 = np.array([1, 1, 4]) / np.sum([1, 1, 4])
    node3.sync(x0, slack, 0.7, 1)
    entropy_gm_draw_constraints(node3)
    # Fill sliding window
    node_stream = NodeStreamFrequency(2, sliding_window_size, 1, x0.shape[0])
    for i in range(sliding_window_size):
        node_stream.set_new_data_point(0, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node3.set_new_data_point(local_vector)
    entropy_gm_draw_constraints(node3)
    assert (not b_inside_safe_zone)
    node_stream.set_new_data_point(1, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node3.set_new_data_point(local_vector)
    entropy_gm_draw_constraints(node3)
    assert (not b_inside_safe_zone)
    node_stream.set_new_data_point(2, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node3.set_new_data_point(local_vector)
    entropy_gm_draw_constraints(node3)
    assert (not b_inside_safe_zone)
    node_stream.set_new_data_point(2, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node3.set_new_data_point(local_vector)
    entropy_gm_draw_constraints(node3)
    assert (not b_inside_safe_zone)
    node_stream.set_new_data_point(2, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node3.set_new_data_point(local_vector)
    entropy_gm_draw_constraints(node3)
    assert b_inside_safe_zone
    node3.sync(x0, slack, 0.7, 1)
    entropy_gm_draw_constraints(node3)

    slack = np.zeros(2)  # No slack

    # 2 dimensional case
    sliding_window_size = 4
    node2 = GmEntropyNode(idx=1, d=2, func_to_monitor=func_entropy)
    x0 = np.array([1, 4]) / np.sum([1, 4])
    node2.sync(x0, slack, 0.5, 0.6)
    entropy_gm_draw_constraints(node2)
    # Fill sliding window
    node_stream = NodeStreamFrequency(2, sliding_window_size, 1, x0.shape[0])
    for i in range(sliding_window_size):
        node_stream.set_new_data_point(0, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node2.set_new_data_point(local_vector)
    entropy_gm_draw_constraints(node2)
    assert (not b_inside_safe_zone)
    node_stream.set_new_data_point(1, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node2.set_new_data_point(local_vector)
    entropy_gm_draw_constraints(node2)
    assert (not b_inside_safe_zone)
    node_stream.set_new_data_point(1, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node2.set_new_data_point(local_vector)
    entropy_gm_draw_constraints(node2)
    assert (not b_inside_safe_zone)
    node_stream.set_new_data_point(1, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node2.set_new_data_point(local_vector)
    entropy_gm_draw_constraints(node2)
    assert b_inside_safe_zone
    node_stream.set_new_data_point(1, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node2.set_new_data_point(local_vector)
    entropy_gm_draw_constraints(node2)
    assert (not b_inside_safe_zone)
    node2.sync(x0, slack, 0.5, 0.6)
    entropy_gm_draw_constraints(node2)


def visualize_variance():
    slack = np.zeros(2)  # No slack
    sliding_window_size = 5
    x0 = np.array([0.5, 2])
    node_idx = 1

    node = GmVarianceNode(idx=node_idx, func_to_monitor=func_variance)
    node.sync(x0, slack, 0.08, 3)
    # Fill sliding window
    node_stream = NodeStreamFirstAndSecondMomentum(2, sliding_window_size, 1, x0.shape[0])
    for i in range(sliding_window_size):
        node_stream.set_new_data_point(0, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node.set_new_data_point(local_vector)
    variance_gm_draw_constraints(node)
    assert (not b_inside_safe_zone)
    node_stream.set_new_data_point(2, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node.set_new_data_point(local_vector)
    variance_gm_draw_constraints(node)
    assert b_inside_safe_zone
    node_stream.set_new_data_point(6, node_idx)
    local_vector = node_stream.get_local_vector(node_idx)
    b_inside_safe_zone = node.set_new_data_point(local_vector)
    variance_gm_draw_constraints(node)
    assert (not b_inside_safe_zone)


if __name__ == "__main__":
    plt.close("all")
    visualize_entropy()
    visualize_variance()
