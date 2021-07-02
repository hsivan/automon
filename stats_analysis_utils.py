import os
import numpy as np
from test_utils import read_config_file
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
# matplotlib.use('tkagg')


def _is_period_approximation_error_above_error_bound(period, real_function_value, error_bound):
    stream_len = len(real_function_value)
    f_x0_sync_points = real_function_value[::period]
    f_x0_at_every_second = f_x0_sync_points.repeat(period)[:stream_len]
    approximation_error = np.abs(real_function_value - f_x0_at_every_second)

    if np.any((approximation_error > error_bound)):
        return True
    return False


def _search_largest_period_under_error_bound(real_function_value, error_bound):
    stream_len = len(real_function_value)
    period = 1

    while period < np.min((500, stream_len)):
        is_above_error_bound = _is_period_approximation_error_above_error_bound(period, real_function_value, error_bound)
        if is_above_error_bound:
            return np.max((1, period - 1))
        else:
            period += 1
    return period


def _get_period_approximation_error(period, real_function_value, num_nodes):
    stream_len = len(real_function_value)
    f_x0_sync_points = real_function_value[::period]
    f_x0_at_every_second = f_x0_sync_points.repeat(period)[:stream_len]
    approximation_error = np.abs(real_function_value - f_x0_at_every_second)

    # Step function: at each sync point the step increases by num_nodes, when each node sends its statistics (no sync message needed in periodic)
    cumulative_msgs = np.array([num_nodes * (i + 1) for i in range(len(f_x0_sync_points))])
    cumulative_msgs = cumulative_msgs.repeat(period)[:stream_len]

    return approximation_error, cumulative_msgs


def _plot_eager_sync_messages(test_folder, data_len, colors, linestyles, oracle_period, periodic_cumulative_msg):
    conf = read_config_file(test_folder)
    num_nodes = conf["num_nodes"]

    cumulative_msgs_broadcast_enabled_suffix = "_cumulative_msgs_broadcast_enabled.csv"
    cumulative_msgs_broadcast_enabled_files = [f for f in os.listdir(test_folder) if f.endswith(cumulative_msgs_broadcast_enabled_suffix)]
    cumulative_msgs_broadcast_enabled_files.sort()
    cumulative_fallback_to_eager_sync_suffix = "_cumulative_fallback_to_eager_sync.csv"
    cumulative_fallback_to_eager_sync_files = [f for f in os.listdir(test_folder) if f.endswith(cumulative_fallback_to_eager_sync_suffix)]
    cumulative_fallback_to_eager_sync_files.sort()

    # Eager sync
    fig, axs = plt.subplots(2, 1)
    for idx, file in enumerate(cumulative_fallback_to_eager_sync_files):
        cumulative_fallback_to_eager_sync = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(cumulative_fallback_to_eager_sync_suffix, "")
        axs[0].plot(np.arange(cumulative_fallback_to_eager_sync.shape[0]),
                    cumulative_fallback_to_eager_sync,
                    label=coordinator_name, color=colors[idx], linestyle=linestyles[idx])
    axs[0].set_title("Cumulative fallback to eager sync")

    # Cumulative messages - broadcast enabled
    for idx, file in enumerate(cumulative_msgs_broadcast_enabled_files):
        cumulative_msgs_broadcast_enabled = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(cumulative_msgs_broadcast_enabled_suffix, "")
        axs[1].plot(np.arange(data_len), cumulative_msgs_broadcast_enabled,
                    label=coordinator_name + "(broadcast)", color=colors[idx], linestyle=linestyles[idx])
    axs[1].plot(np.arange(data_len), periodic_cumulative_msg,
                label="Periodic oracle " + str(oracle_period) + " (broadcast)", color=colors[-2], linestyle=linestyles[-1])
    # Centralization line - every node sends its statistics every second
    axs[1].plot(np.arange(data_len),
                np.arange(1, data_len + 1) * num_nodes,
                label="Centralization", color="black", linestyle='-')
    axs[1].set_title("Cumulative messages - broadcast enabled")

    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    fig.savefig(test_folder + "/cumulative_fallback_to_eager_sync.pdf")
    plt.close(fig)


def plot_figures(test_folder):
    conf = read_config_file(test_folder)
    error_bound = conf["error_bound"]
    num_nodes = conf["num_nodes"]

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    real_function_value_file_suffix = "_real_function_value.csv"
    real_function_value_files = [f for f in os.listdir(test_folder) if f.endswith(real_function_value_file_suffix)]
    function_approximation_error_file_suffix = "_function_approximation_error.csv"
    function_approximation_error_files = [f for f in os.listdir(test_folder) if f.endswith(function_approximation_error_file_suffix)]
    function_approximation_error_files.sort()
    cumulative_msgs_broadcast_disabled_suffix = "_cumulative_msgs_broadcast_disabled.csv"
    cumulative_msgs_broadcast_disabled_files = [f for f in os.listdir(test_folder) if f.endswith(cumulative_msgs_broadcast_disabled_suffix)]
    cumulative_msgs_broadcast_disabled_files.sort()

    # Real function value
    real_function_value = np.genfromtxt(test_folder + "/" + real_function_value_files[0])  # Need only one, as they are all the same
    data_len = len(real_function_value)
    axs[0].plot(np.arange(data_len), real_function_value, label="real f val", color="blue")
    axs[0].fill_between(np.arange(data_len), real_function_value - error_bound, real_function_value + error_bound, facecolor='b', alpha=0.2)
    axs[0].set_title("Real function value and thresholds (U,L)")

    period = _search_largest_period_under_error_bound(real_function_value, error_bound)
    periodic_approximation_error, periodic_cumulative_msg = _get_period_approximation_error(period, real_function_value, num_nodes)

    # Approximation error
    colors = ["blue", "orange", "green", "red"]
    linestyles = ['-', '--', ':']
    for idx, file in enumerate(function_approximation_error_files):
        function_approximation_error = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(function_approximation_error_file_suffix, "")
        axs[1].plot(np.arange(data_len), function_approximation_error,
                    label=coordinator_name, color=colors[idx], linestyle=linestyles[idx])
    axs[1].plot(np.arange(data_len), periodic_approximation_error,
                label="Periodic oracle " + str(period), color=colors[-2], linestyle=linestyles[-1])
    axs[1].set_title("Approximation error")

    # Cumulative messages - broadcast disabled
    auto_mon_num_messages = 0
    for idx, file in enumerate(cumulative_msgs_broadcast_disabled_files):
        cumulative_msgs_broadcast_disabled = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(cumulative_msgs_broadcast_disabled_suffix, "")
        axs[2].plot(np.arange(data_len), cumulative_msgs_broadcast_disabled,
                    label=coordinator_name, color=colors[idx], linestyle=linestyles[idx])
        if "AutoMon" in coordinator_name:
            auto_mon_num_messages = cumulative_msgs_broadcast_disabled[-1]
    axs[2].plot(np.arange(data_len), periodic_cumulative_msg,
                label="Periodic oracle " + str(period), color=colors[-2], linestyle=linestyles[-1])

    # Find the period that provides auto_mon_num_messages messages.
    if auto_mon_num_messages == 0:
        period_equiv = len(real_function_value) - 1
    else:
        period_equiv = np.max((1, int(len(real_function_value) * num_nodes / auto_mon_num_messages)))
    periodic_equiv_approximation_error, periodic_equiv_cumulative_msg = _get_period_approximation_error(
        period_equiv, real_function_value, num_nodes)
    axs[1].plot(np.arange(data_len), periodic_equiv_approximation_error,
                label="Periodic equiv " + str(period_equiv), color=colors[-1], linestyle=linestyles[-1])
    axs[2].plot(np.arange(data_len), periodic_equiv_cumulative_msg,
                label="Periodic equiv " + str(period_equiv), color=colors[-1], linestyle=linestyles[-1])

    # Centralization line - every node sends its statistics every second
    axs[2].plot(np.arange(data_len), np.arange(1, data_len + 1) * num_nodes,
                label="Centralization", color="black", linestyle='-')
    axs[2].set_title("Cumulative messages")
    axs[2].set_yscale('log')

    # Error bound
    axs[1].hlines(error_bound, xmin=0, xmax=data_len - 1, linestyles='dashed', color="black", label="error bound")

    axs[0].legend(bbox_to_anchor=(1.225, 1.1))
    axs[1].legend(bbox_to_anchor=(1.0, 1.1))
    axs[2].legend(bbox_to_anchor=(1.0, 1.1))
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    fig.savefig(test_folder + "/func_val_and_approx_error.pdf")

    plt.close(fig)
    _plot_eager_sync_messages(test_folder, data_len, colors, linestyles, period, periodic_cumulative_msg)


def get_violations_per_neighborhood_size(test_folder):
    # Returns array of neighborhood sizes, and another two arrays - one with number of neighborhood violations and one
    # with number of safe zone violations for every neighborhood size in the neighborhood_sizes array
    neighborhood_sizes = []
    safe_zone_violations_arr = []
    neighborhood_violations_arr = []
    sub_test_folders = list(filter(lambda x: os.path.isdir(os.path.join(test_folder, x)), os.listdir(test_folder)))
    sub_test_folders.sort()
    sub_test_folders = [test_folder + "/" + sub_folder for sub_folder in sub_test_folders]

    for sub_test_folder in sub_test_folders:
        conf = read_config_file(sub_test_folder)
        neighborhood_size = conf["neighborhood_size"]

        try:
            results_file = open(sub_test_folder + "/results.txt", 'r')
            results = results_file.read()
            results = results.split("\n")
            results = [result for result in results if
                       "AutoCB" in result or "AutoMon" in result or "caused by local vector outside safe zone" in result or "caused by local vector outside domain" in result]
            for idx, result in enumerate(results):
                if "AutoCB" in result or "AutoMon" in result:
                    safe_zone_violations = int(results[idx + 1].split(" ")[-1])
                    neighborhood_violations = int(results[idx + 2].split(" ")[-1])
            neighborhood_sizes += [neighborhood_size]
            safe_zone_violations_arr += [safe_zone_violations]
            neighborhood_violations_arr += [neighborhood_violations]
        except Exception as e:
            print(e)
            neighborhood_sizes += [neighborhood_size]
            safe_zone_violations_arr += [safe_zone_violations_arr[-1]]
            neighborhood_violations_arr += [neighborhood_violations_arr[-1]]

    return neighborhood_sizes, neighborhood_violations_arr, safe_zone_violations_arr


def plot_impact_of_neighborhood_size_on_violations(test_folder):
    neighborhood_sizes, neighborhood_violations_arr, safe_zone_violations_arr = get_violations_per_neighborhood_size(test_folder)

    fig, ax = plt.subplots()
    ax.stackplot(neighborhood_sizes, neighborhood_violations_arr, safe_zone_violations_arr, labels=['neighborhood', 'safe zone'])
    ax.set_xlabel('neighborhood size around $x_0$')
    ax.set_ylabel('num violations')
    ax.set_title('Impact of neighborhood size on violations')
    ax.legend()

    fig.savefig(test_folder + "/impact_of_neighborhood_size_on_violations.pdf")
    plt.close(fig)


def get_optimal_neighborhood_size(test_folder):
    neighborhood_sizes, neighborhood_violations_arr, safe_zone_violations_arr = get_violations_per_neighborhood_size(test_folder)
    violations = np.sum((neighborhood_violations_arr, safe_zone_violations_arr), axis=0)
    min_idx = np.argmin(violations)
    return neighborhood_sizes[min_idx]


def get_tuned_neighborhood_size(test_folder):
    with open(test_folder + "/tuned_neighborhood_size.txt", "r") as f:
        tuned_neighborhood_size = f.read()
        tuned_neighborhood_size = float(tuned_neighborhood_size.split(" ")[1].replace("/n", ""))
    return tuned_neighborhood_size


def get_neighborhood_size_error_bound_connection(parent_test_folder):
    sub_test_folders = list(filter(lambda x: os.path.isdir(os.path.join(parent_test_folder, x)), os.listdir(parent_test_folder)))
    sub_test_folders.sort()
    test_folders = [parent_test_folder + "/" + sub_folder for sub_folder in sub_test_folders]
    optimal_neighborhood_sizes = np.zeros(len(test_folders))
    error_bounds = np.zeros(len(test_folders))
    tuned_neighborhood_sizes = np.zeros(len(test_folders))

    for idx, test_folder in enumerate(test_folders):
        conf = read_config_file(test_folder)
        error_bound = conf["error_bound"]
        neighborhood_size = get_optimal_neighborhood_size(test_folder)
        optimal_neighborhood_sizes[idx] = neighborhood_size
        error_bounds[idx] = error_bound
        tuned_neighborhood_size = get_tuned_neighborhood_size(test_folder)
        tuned_neighborhood_sizes[idx] = tuned_neighborhood_size

    return error_bounds, optimal_neighborhood_sizes, tuned_neighborhood_sizes


def plot_neighborhood_size_error_bound_connection(parent_test_folder):
    sub_test_folders = list(filter(lambda x: os.path.isdir(os.path.join(parent_test_folder, x)), os.listdir(parent_test_folder)))
    sub_test_folders.sort()
    test_folders = [parent_test_folder + "/" + sub_folder for sub_folder in sub_test_folders]
    test_folders.sort()
    optimal_neighborhood_sizes = np.zeros(len(test_folders))
    error_bounds = np.zeros(len(test_folders))

    for idx, test_folder in enumerate(test_folders):
        conf = read_config_file(test_folder)
        error_bound = conf["error_bound"]
        optimal_neighborhood_size = get_optimal_neighborhood_size(test_folder)
        optimal_neighborhood_sizes[idx] = optimal_neighborhood_size
        error_bounds[idx] = error_bound

    fig, ax = plt.subplots()
    ax.plot(error_bounds, optimal_neighborhood_sizes, linestyle='-', marker='x')
    ax.set_xlabel('error bound')
    ax.set_ylabel('neighborhood size around $x_0$')
    ax.set_title('neighborhood size - error bound connection')
    ax.legend()

    fig.savefig(parent_test_folder + "/neighborhood_size_error_bound_connection.pdf")
    plt.close(fig)
    return error_bounds, optimal_neighborhood_sizes


def plot_neighborhood_size_error_bound_connection_avg(parent_test_folder):
    experiment_folders = list(
        filter(lambda x: os.path.isdir(os.path.join(parent_test_folder, x)), os.listdir(parent_test_folder)))
    experiment_folders.sort()
    experiment_folders = [parent_test_folder + "/" + sub_folder for sub_folder in experiment_folders]

    fig, ax = plt.subplots()

    optimal_neighborhood_sizes_experiments = []
    for experiment in experiment_folders:
        error_bounds, optimal_neighborhood_sizes = plot_neighborhood_size_error_bound_connection(experiment)
        optimal_neighborhood_sizes_experiments.append(optimal_neighborhood_sizes)
    optimal_neighborhood_sizes_mean = np.mean(optimal_neighborhood_sizes_experiments, axis=0)
    optimal_neighborhood_sizes_std = np.std(optimal_neighborhood_sizes_experiments, axis=0)

    ax.plot(error_bounds, optimal_neighborhood_sizes_mean, linestyle='-', marker='x')
    ax.fill_between(error_bounds, optimal_neighborhood_sizes_mean - optimal_neighborhood_sizes_std, optimal_neighborhood_sizes_mean + optimal_neighborhood_sizes_std, alpha=.3)
    ax.set_xlabel('error bound')
    ax.set_ylabel('Mean neighborhood size around $x_0$')
    ax.set_title('neighborhood size - error bound connection')
    ax.legend()

    fig.savefig(parent_test_folder + "/neighborhood_size_error_bound_connection_avg.pdf")
    plt.close(fig)
