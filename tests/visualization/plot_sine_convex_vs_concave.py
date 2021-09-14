import autograd.numpy as np
from autograd import hessian
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from scipy.optimize import brentq
from autograd import grad
from tests.visualization.utils import get_figsize


def func_sine(X):
    return np.sin(X)


def plot_admissible_region(X0, xlim, ylim, func_to_convex, l_thresh, u_thresh, left_range, right_range):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    # textwidth is 506.295, columnwidth is 241.14749
    fig = plt.figure(figsize=get_figsize(columnwidth=506.295, wf=0.32, hf=0.42))

    ax = fig.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Make data
    X = np.arange(xlim[0], xlim[1], 0.01)
    Y = func_to_convex(X)
    # Plot the sine in the domain
    ax.plot(X, Y, label="$sin(x)$", color="black", linewidth=0.9)

    ax.plot(X0, func_to_convex(X0), 'o', label="$x_0$", color="black", markersize=3)

    ax.axvspan(xmin=left_range, xmax=right_range, edgecolor="none", facecolor="grey", alpha=0.25)
    plt.xticks([left_range, right_range])

    ax.hlines(y=u_thresh, xmin=xlim[0], xmax=xlim[1], colors='tab:blue', linestyles="--", label='$U$', linewidth=0.9)
    ax.hlines(y=l_thresh, xmin=xlim[0], xmax=xlim[1], colors='tab:red', linestyles="--", label='$L$', linewidth=0.9)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.34), ncol=4, columnspacing=1, handlelength=1.7, frameon=False)
    plt.subplots_adjust(top=0.82, bottom=0.18, left=0.09, right=0.96, hspace=0.08, wspace=0.2)

    fig_file_name = "sine_admissible_region.pdf"
    plt.savefig(fig_file_name)
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_2d_convex_diff(X0, xlim, ylim, func_to_convex, l_thresh, u_thresh, func_g_convex, func_h_convex, func_g_minus_l_tangent,
                        left_range, right_range):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    # textwidth is 506.295, columnwidth is 241.14749
    fig = plt.figure(figsize=get_figsize(columnwidth=506.295, wf=0.32, hf=0.42))

    ax = fig.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Make data
    X = np.arange(xlim[0], xlim[1], 0.01)
    Y = func_to_convex(X)
    # Plot the sine in the domain
    ax.plot(X, Y, color="black", linewidth=0.9)

    ax.plot(X0, func_to_convex(X0), 'o', color="black", markersize=3)

    g_convex_val = func_g_convex(X)
    h_convex_val = func_h_convex(X)
    ax.plot(X, g_convex_val, label=r"$\breve{g}(x)$", color="tab:orange", linewidth=0.9)
    ax.plot(X, h_convex_val, label=r"$\breve{h}(x)$", color="tab:green", linewidth=0.9)

    g_minus_l_tangent_val = func_g_minus_l_tangent(X)

    ax.fill_between(X, g_convex_val, u_thresh, where=g_convex_val < u_thresh, color='orange', alpha=0.5, interpolate=True)
    ax.fill_between(X, h_convex_val, g_minus_l_tangent_val, where=h_convex_val < g_minus_l_tangent_val, color='green', alpha=0.3, interpolate=True)

    ax.axvline(x=left_range, ymin=-1, ymax=1, linestyle=":", color="grey", linewidth=0.8)
    ax.axvline(x=right_range, ymin=-1, ymax=1, linestyle=":", color="grey", linewidth=0.8)
    plt.xticks([left_range, right_range])

    ax.hlines(y=u_thresh, xmin=xlim[0], xmax=xlim[1], colors='tab:blue', linestyles="--", linewidth=0.9)
    ax.hlines(y=l_thresh, xmin=xlim[0], xmax=xlim[1], colors='tab:red', linestyles="--", linewidth=0.9)

    ax.plot(X, g_minus_l_tangent_val, label=r"$f(x_0)+\nabla f(x_0)^T (x-x_0) - L$", color="grey", linestyle="-.", linewidth=0.9)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.36), ncol=3, columnspacing=0.8, handlelength=1.7, frameon=False, handletextpad=0.8)
    plt.subplots_adjust(top=0.82, bottom=0.18, left=0.09, right=0.96, hspace=0.08, wspace=0.2)

    fig_file_name = "sine_convex_diff.pdf"
    plt.savefig(fig_file_name)
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_2d_concave_diff(X0, xlim, ylim, func_to_concave, l_thresh, u_thresh, func_g_concave, func_h_concave, func_g_minus_u_tangent,
                         left_range, right_range):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    # textwidth is 506.295, columnwidth is 241.14749
    fig = plt.figure(figsize=get_figsize(columnwidth=506.295, wf=0.32, hf=0.42))

    ax = fig.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Make data
    X = np.arange(xlim[0], xlim[1], 0.01)
    Y = func_to_concave(X)
    # Plot the sine in the domain
    ax.plot(X, Y, color="black", linewidth=0.9)

    ax.plot(X0, func_to_concave(X0), 'o', color="black", markersize=3)

    g_concave_val = func_g_concave(X)
    h_concave_val = func_h_concave(X)
    ax.plot(X, g_concave_val, label="$\hat{g}(x)$", color="tab:green", linewidth=0.9)
    ax.plot(X, h_concave_val, label="$\hat{h}(x)$", color="tab:orange", linewidth=0.9)

    g_minus_u_tangent_val = func_g_minus_u_tangent(X)

    ax.fill_between(X, g_concave_val, l_thresh, where=g_concave_val > l_thresh, color='green', alpha=0.5, interpolate=True)
    ax.fill_between(X, h_concave_val, g_minus_u_tangent_val, where=h_concave_val > g_minus_u_tangent_val, color='orange', alpha=0.3, interpolate=True)

    ax.axvline(x=left_range, ymin=-1, ymax=1, linestyle=":", color="grey", linewidth=0.8)
    ax.axvline(x=right_range, ymin=-1, ymax=1, linestyle=":", color="grey", linewidth=0.8)
    plt.xticks([left_range, right_range])

    ax.hlines(y=u_thresh, xmin=xlim[0], xmax=xlim[1], colors='tab:blue', linestyles="--", linewidth=0.9)
    ax.hlines(y=l_thresh, xmin=xlim[0], xmax=xlim[1], colors='tab:red', linestyles="--", linewidth=0.9)

    ax.plot(X, g_minus_u_tangent_val, label=r"$f(x_0) +\nabla f(x_0)^T (x-x_0) - U$", color="grey", linestyle="-.", linewidth=0.9)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.38), ncol=3, columnspacing=0.8, handlelength=1.7, frameon=False, handletextpad=0.8)
    plt.subplots_adjust(top=0.82, bottom=0.18, left=0.09, right=0.96, hspace=0.08, wspace=0.2)

    fig_file_name = "sine_concave_diff.pdf"
    plt.savefig(fig_file_name)
    plt.close(fig)
    rcParams.update(rcParamsDefault)


# Minimize this function over x in a specific neighborhood around X0
def func_min_eigenvalue(x, args):
    hess = args
    eigenvalues, eigenvector = np.linalg.eig(hess(x))
    min_eigenvalue = np.min(eigenvalues)
    return min_eigenvalue


# Maximize this function over x in a specific neighborhood around X0
def func_max_eigenvalue(x, args):
    hess = args
    eigenvalues, eigenvector = np.linalg.eig(hess(x))
    max_eigenvalue = np.max(eigenvalues)
    return -1.0 * max_eigenvalue


def func_min_max_eigenvalues(func_to_convex, X0, domain):
    hess = hessian(func_to_convex)
    sol_min = minimize(func_min_eigenvalue, X0, args=hess, bounds=domain)
    sol_max = minimize(func_max_eigenvalue, X0, args=hess, bounds=domain)
    min_eigenvalue = sol_min.fun
    minimizing_point = sol_min.x
    max_eigenvalue = -1.0 * sol_max.fun
    maximizing_point = sol_max.x
    return min_eigenvalue, minimizing_point, max_eigenvalue, maximizing_point


def find_min_and_max_eigenvalues(func_to_convex, domain):
    min_eigenvalue = np.inf
    max_eigenvalue = -np.inf
    # Start the optimization process from multiple points in the domain, then choose the max and min
    rand_start_points = np.random.uniform(domain[0][0], domain[0][1], (10, len(domain)))
    for start_point in rand_start_points:
        start_point = np.array(start_point, dtype=np.float32)
        min_eigenvalue_temp, minimizing_point, max_eigenvalue_temp, maximizing_point = func_min_max_eigenvalues(func_to_convex, start_point, domain)
        min_eigenvalue = np.minimum(min_eigenvalue, min_eigenvalue_temp)
        max_eigenvalue = np.maximum(max_eigenvalue, max_eigenvalue_temp)

    assert (min_eigenvalue <= max_eigenvalue)
    print("max_eigenvalue:", max_eigenvalue)
    print("min_eigenvalue:", min_eigenvalue)
    return min_eigenvalue, max_eigenvalue


def admissible_region(X0, func_to_convex, xlim=None, ylim=None):
    l_thresh = func_to_convex(X0) - 0.2
    u_thresh = func_to_convex(X0) + 0.2
    func = lambda x: func_to_convex(x) - l_thresh
    lower_thresh_root_left, lower_thresh_root_right = brentq(func, X0 - 2, X0), brentq(func, X0, X0 + 2)

    plot_admissible_region(X0, xlim, ylim, func_to_convex, l_thresh, u_thresh, lower_thresh_root_left, lower_thresh_root_right)


def convex_diff(X0, min_eigenvalue, func_to_convex, xlim=None, ylim=None):
    l_thresh = func_to_convex(X0) - 0.2
    u_thresh = func_to_convex(X0) + 0.2
    search_roots_distance = 10

    eig = min_eigenvalue
    g_convex = lambda x: func_to_convex(x) + 0.5 * np.abs(eig) * (x - X0) * (x - X0)
    h_convex = lambda x: 0.5 * np.abs(eig) * (x - X0) * (x - X0)

    grad_func_to_convex = grad(func_to_convex)
    g_minus_l_tangent = lambda x: func_to_convex(X0) + grad_func_to_convex(X0) * (x - X0) - l_thresh

    # Condition 1: g(x) < U.
    # Check where g(x) = U and write in title
    func = lambda x: g_convex(x) - u_thresh
    upper_thresh_root_left, upper_thresh_root_right = brentq(func, X0 - search_roots_distance, X0), brentq(func, X0, X0 + search_roots_distance)
    assert upper_thresh_root_left <= upper_thresh_root_right, str(upper_thresh_root_left) + "," + str(upper_thresh_root_right)
    upper_safe_zone_size = upper_thresh_root_right - upper_thresh_root_left

    # Condition 2: Tangent g(x)-L bigger than h(x)
    # Check where Tangent g(x)-L = h(x) and write in figure title
    func = lambda x: g_minus_l_tangent(x) - h_convex(x)
    lower_thresh_root_left, lower_thresh_root_right = brentq(func, X0 - search_roots_distance, X0), brentq(func, X0, X0 + search_roots_distance)
    assert lower_thresh_root_left <= lower_thresh_root_right, str(lower_thresh_root_left) + "," + str(lower_thresh_root_right)
    lower_safe_zone_size = lower_thresh_root_right - lower_thresh_root_left

    if upper_safe_zone_size == 0 or lower_safe_zone_size == 0:
        safe_zone_size = 0
    else:
        safe_zone_size = np.minimum(upper_thresh_root_right, lower_thresh_root_right) - np.maximum(upper_thresh_root_left, lower_thresh_root_left)
    assert safe_zone_size >= 0, str(safe_zone_size)

    plot_2d_convex_diff(X0, xlim, ylim, func_to_convex, l_thresh, u_thresh, g_convex, h_convex, g_minus_l_tangent,
                        np.maximum(lower_thresh_root_left, upper_thresh_root_left),
                        np.minimum(lower_thresh_root_right, upper_thresh_root_right))

    return safe_zone_size, upper_safe_zone_size, lower_safe_zone_size


def concave_diff(X0, max_eigenvalue, func_to_concave, xlim=None, ylim=None):
    l_thresh = func_to_concave(X0) - 0.2
    u_thresh = func_to_concave(X0) + 0.2
    search_roots_distance = 10

    g_concave = lambda x: func_to_concave(x) - 0.5 * max_eigenvalue * (x - X0) * (x - X0)
    h_concave = lambda x: -0.5 * max_eigenvalue * (x - X0) * (x - X0)

    grad_func_to_convex = grad(func_to_concave)
    g_minus_u_tangent = lambda x: func_to_concave(X0) + grad_func_to_convex(X0) * (x - X0) - u_thresh

    # Condition 1: g(x) > L.
    # Check where g(x) = L and write in title
    func = lambda x: g_concave(x) - l_thresh
    lower_thresh_root_left, lower_thresh_root_right = brentq(func, X0 - search_roots_distance, X0), brentq(func, X0, X0 + search_roots_distance)
    assert lower_thresh_root_left <= lower_thresh_root_right, str(lower_thresh_root_left) + "," + str(lower_thresh_root_right)
    lower_safe_zone_size = lower_thresh_root_right - lower_thresh_root_left

    # Condition 2: Tangent g(x)-U smaller than h(x)
    # Check where Tangent g(x)-U = h(x) and write in figure title
    func = lambda x: g_minus_u_tangent(x) - h_concave(x)
    upper_thresh_root_left, upper_thresh_root_right = brentq(func, X0 - search_roots_distance, X0), brentq(func, X0, X0 + search_roots_distance)
    assert upper_thresh_root_left <= upper_thresh_root_right, str(upper_thresh_root_left) + "," + str(upper_thresh_root_right)
    upper_safe_zone_size = upper_thresh_root_right - upper_thresh_root_left

    if upper_safe_zone_size == 0 or lower_safe_zone_size == 0:
        safe_zone_size = 0
    else:
        safe_zone_size = np.minimum(upper_thresh_root_right, lower_thresh_root_right) - np.maximum(
            upper_thresh_root_left, lower_thresh_root_left)
    assert safe_zone_size >= 0, str(safe_zone_size)

    plot_2d_concave_diff(X0, xlim, ylim, func_to_concave, l_thresh, u_thresh, g_concave, h_concave, g_minus_u_tangent,
                         np.maximum(lower_thresh_root_left, upper_thresh_root_left),
                         np.minimum(lower_thresh_root_right, upper_thresh_root_right))

    return safe_zone_size, upper_safe_zone_size, lower_safe_zone_size


if __name__ == "__main__":
    # Figure 1
    X0 = 0.5 * np.pi
    domain = ((X0 - 3, X0 + 3),)
    xlim = [X0 - 2.1, X0 + 2.1]
    ylim = [-0.5, 1.5]
    X0 = np.array([X0])

    min_eigenvalue, max_eigenvalue = find_min_and_max_eigenvalues(func_sine, domain)
    # Plot the concave and convex diffs
    convex_diff(X0, min_eigenvalue, func_sine, xlim=xlim, ylim=ylim)
    concave_diff(X0, max_eigenvalue, func_sine, xlim=xlim, ylim=ylim)
    admissible_region(X0, func_sine, xlim=xlim, ylim=ylim)
