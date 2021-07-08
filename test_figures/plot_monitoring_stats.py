import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import numpy as np
from test_figures.plot_error_communication_tradeoff import get_period_approximation_error
from test_figures.plot_figures_utils import get_figsize
from test_utils import read_config_file
from stats_analysis_utils import _search_largest_period_under_error_bound
import os


def plot_monitoring_stats(test_folder, func_name):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 6})
    rcParams.update({'font.size': 7})

    conf = read_config_file(test_folder)
    error_bound = conf["error_bound"]
    num_nodes = conf["num_nodes"]

    fig, axs = plt.subplots(3, 1, figsize=get_figsize(hf=1.0), sharex=True)
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
    axs[0].plot(np.arange(0, data_len), real_function_value, label=r'$f \pm T$', color="tab:blue", linewidth=1)
    axs[0].fill_between(np.arange(0, data_len), real_function_value - error_bound, real_function_value + error_bound, facecolor='tab:blue', alpha=0.3)
    axs[0].set_ylabel(r'$f$')

    period = _search_largest_period_under_error_bound(real_function_value, error_bound)
    periodic_approximation_error, periodic_cumulative_msg = get_period_approximation_error(period, real_function_value, num_nodes, 0)

    # Approximation error
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    linestyles = ['-', '-.', ':']
    for idx, file in enumerate(function_approximation_error_files):
        function_approximation_error = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(function_approximation_error_file_suffix, "")
        axs[1].plot(np.arange(0, data_len), function_approximation_error, label=coordinator_name, color=colors[idx], linestyle=linestyles[idx], linewidth=1)
    axs[1].plot(np.arange(0, data_len), periodic_approximation_error, label="Periodic oracle " + str(period), color=colors[-2], linestyle=linestyles[-1], linewidth=1)
    axs[1].set_ylabel("Approx. error")

    # Cumulative messages - broadcast disabled
    for idx, file in enumerate(cumulative_msgs_broadcast_disabled_files):
        cumulative_msgs_broadcast_disabled = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(cumulative_msgs_broadcast_disabled_suffix, "")
        axs[2].plot(np.arange(0, data_len), cumulative_msgs_broadcast_disabled, label=coordinator_name, color=colors[idx], linestyle=linestyles[idx], linewidth=1)
    axs[2].plot(np.arange(0, data_len), periodic_cumulative_msg, label="Periodic oracle", color=colors[-2], linestyle=linestyles[-1], linewidth=1)

    # Centralization line - every node sends its statistics every second
    axs[2].plot(np.arange(0, data_len), np.arange(1, data_len + 1) * num_nodes, label="Centralization", color="black", linestyle='--', linewidth=1)
    axs[2].set_ylabel("Cumul. messages")
    axs[2].set_xlabel("rounds")
    axs[2].set_yscale('log')

    # Error bound
    axs[1].hlines(error_bound, xmin=0, xmax=data_len - 1, linestyles='dashed', color="black", label="error bound")

    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[0].yaxis.set_label_coords(-0.13, 0.5)
    axs[1].yaxis.set_label_coords(-0.13, 0.5)
    axs[2].yaxis.set_label_coords(-0.13, 0.5)
    axs[0].spines['bottom'].set_linewidth(0.5)
    axs[0].spines['left'].set_linewidth(0.5)
    axs[0].tick_params(width=0.5)
    axs[1].spines['bottom'].set_linewidth(0.5)
    axs[1].spines['left'].set_linewidth(0.5)
    axs[1].tick_params(width=0.5)
    axs[2].spines['bottom'].set_linewidth(0.5)
    axs[2].spines['left'].set_linewidth(0.5)
    axs[2].tick_params(width=0.5)

    axs[0].legend(loc="upper right", frameon=False, framealpha=0)
    axs[1].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.3))
    axs[2].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.3))
    plt.subplots_adjust(top=0.98, bottom=0.11, left=0.15, right=0.99, hspace=0.4)
    if func_name == "kld":
        axs[1].set_ylim(top=0.08)
        plt.subplots_adjust(top=0.98, bottom=0.11, left=0.16, right=0.99, hspace=0.4)
    fig.savefig("func_val_and_approx_error_" + func_name + ".pdf")

    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    test_folder = "../test_results/results_compare_methods_kld_air_quality_2021-07-09_11-15-23"
    plot_monitoring_stats(test_folder, "kld_air_quality")
