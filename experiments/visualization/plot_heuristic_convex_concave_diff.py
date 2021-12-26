import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import numpy as np
from experiments.visualization.visualization_utils import get_figsize, get_function_value_offset
from test_utils.test_utils import read_config_file
import os


def plot_monitoring_stats_sine(test_folder):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 6})
    rcParams.update({'font.size': 7})

    conf = read_config_file(test_folder)
    error_bound = conf["error_bound"]
    offset = get_function_value_offset(test_folder)

    fig, axs = plt.subplots(3, 1, figsize=get_figsize(hf=1.0), sharex=True)
    real_function_value_file_suffix = "_real_function_value.csv"
    real_function_value_files = [f for f in os.listdir(test_folder) if f.endswith(real_function_value_file_suffix)]
    function_approximation_error_file_suffix = "_function_approximation_error.csv"
    function_approximation_error_files = [f for f in os.listdir(test_folder) if
                                          f.endswith(function_approximation_error_file_suffix)]
    function_approximation_error_files.sort()
    cumulative_msgs_broadcast_disabled_suffix = "_cumulative_msgs_broadcast_disabled.csv"
    cumulative_msgs_broadcast_disabled_files = [f for f in os.listdir(test_folder) if
                                                f.endswith(cumulative_msgs_broadcast_disabled_suffix)]
    cumulative_msgs_broadcast_disabled_files.sort()

    # Real function value
    real_function_value = np.genfromtxt(
        test_folder + "/" + real_function_value_files[0])  # Need only one, as they are all the same
    data_len = len(real_function_value) - offset
    start_iteration = offset
    end_iteration = offset + data_len
    axs[0].plot(np.arange(start_iteration, end_iteration), real_function_value[offset:], label=r'$f \pm T$', color="blue", linewidth=1)
    axs[0].fill_between(np.arange(start_iteration, end_iteration),
                        real_function_value[offset:] - error_bound,
                        real_function_value[offset:] + error_bound, facecolor='b', alpha=0.2)
    axs[0].set_ylabel(r'$f$')

    # Approximation error
    colors = ["orange", "green", "blue"]
    linestyles = ['--', ':', '-']
    for idx, file in enumerate(function_approximation_error_files):
        function_approximation_error = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(function_approximation_error_file_suffix, "")
        if "Concave" in coordinator_name:
            label = "concave diff"
        elif "Convex" in coordinator_name:
            label = "convex diff"
        else:
            label = "auto heuristic"
        axs[1].plot(np.arange(start_iteration, end_iteration), function_approximation_error[offset:],
                    label=label, color=colors[idx], linestyle=linestyles[idx], linewidth=1)
    axs[1].set_ylabel("Approx. error")

    # Cumulative messages - broadcast disabled
    for idx, file in enumerate(cumulative_msgs_broadcast_disabled_files):
        cumulative_msgs_broadcast_disabled = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(cumulative_msgs_broadcast_disabled_suffix, "")
        if "Concave" in coordinator_name:
            label = "concave diff"
        elif "Convex" in coordinator_name:
            label = "convex diff"
        else:
            label = "auto heuristic"
        axs[2].plot(np.arange(start_iteration, end_iteration), cumulative_msgs_broadcast_disabled[offset:],
                    label=label, color=colors[idx], linestyle=linestyles[idx], linewidth=1)

    # Centralization line - every node sends its statistics every second
    axs[2].set_ylabel("Cumul. messages")
    axs[2].set_xlabel("time (seconds)")

    # Error bound
    axs[1].hlines(error_bound, xmin=start_iteration, xmax=end_iteration - 1, linestyles='dashed', color="black", label="error bound")

    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right", ncol=2)
    axs[2].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.3))
    plt.subplots_adjust(top=0.98, bottom=0.11, left=0.15, right=0.99, hspace=0.4)
    fig.savefig("func_val_and_approx_error_sine.pdf")

    rcParams.update(rcParamsDefault)


def get_num_violations(test_folder):
    safe_zone_violations, domain_violations = np.inf, np.inf
    results_file = open(test_folder + "/results.txt", 'r')
    results = results_file.read()
    results = results.split("\n")
    results = [result for result in results if "AutoMon" in result or "caused by local vector outside safe zone" in result or "caused by local vector outside domain" in result]
    for idx, result in enumerate(results):
        if "AutoMon" in result:
            safe_zone_violations = int(results[idx + 1].split(" ")[-1])
            domain_violations = int(results[idx + 2].split(" ")[-1])
    return safe_zone_violations, domain_violations


def plot_violations_histogram(convex_diff_folder, concave_diff_folder, heuristic_folder):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 6})
    rcParams.update({'font.size': 7})

    convex_diff_sz_violations, _ = get_num_violations(convex_diff_folder)
    concave_diff_sz_violations, _ = get_num_violations(concave_diff_folder)
    heuristic_diff_sz_violations, _ = get_num_violations(heuristic_folder)

    fig = plt.figure(figsize=get_figsize(hf=0.4))
    ax = fig.gca()
    ax.set_ylabel('#violations')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    violations = [heuristic_diff_sz_violations, convex_diff_sz_violations, concave_diff_sz_violations]
    ax.bar(["auto heuristic", "convex diff", "concave diff"], violations)
    for i, v in enumerate(violations):
        ax.text(i-0.1, v + 3, str(v), color='blue', fontweight='bold')

    plt.subplots_adjust(top=0.97, bottom=0.17, left=0.14, right=0.99)
    fig.savefig("violations_histogram_sine.pdf")

    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    test_folder = "./sine_monitoring/results_compare_methods_sine_merged_all_methods_2021-01-23_17-36-55/"
    plot_monitoring_stats_sine(test_folder)

    convex_diff_folder = "./sine_monitoring/results_compare_methods_sine_2021-01-23_17-37-31/"
    concave_diff_folder = "./sine_monitoring/results_compare_methods_sine_2021-01-23_17-38-20/"
    heuristic_folder = "./sine_monitoring/results_compare_methods_sine_2021-01-23_17-36-55/"
    plot_violations_histogram(convex_diff_folder, concave_diff_folder, heuristic_folder)
