import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import numpy as np
from tests.visualization.utils import get_figsize, reformat_large_tick_values, get_function_value_offset
from utils.test_utils import read_config_file
import os
import matplotlib.ticker as tick


def plot_monitoring_stats_graph_and_barchart(test_folder, func_name, relative_folder='./'):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    conf = read_config_file(test_folder)
    error_bound = conf["error_bound"]
    num_nodes = conf["num_nodes"]
    offset = get_function_value_offset(test_folder)

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
    real_function_value = np.genfromtxt(test_folder + "/" + real_function_value_files[0])  # Need only one, as they are all the same
    data_len = len(real_function_value) - offset
    start_iteration = offset
    end_iteration = offset + data_len

    ############# Figure of -x_1^2 + x_2^2 function value +-error bound around it #############

    fig = plt.figure(figsize=get_figsize(wf=0.5))
    ax = fig.add_subplot(111)

    ax.plot(np.arange(start_iteration, end_iteration), real_function_value[offset:],
            label=r'$f \pm T$', color="black", linewidth=0.5)
    ax.fill_between(np.arange(start_iteration, end_iteration),
                    real_function_value[offset:] - error_bound,
                    real_function_value[offset:] + error_bound, facecolor='tab:blue', alpha=0.3)
    ax.set_ylabel('f')
    ax.set_xlabel('rounds')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    plt.subplots_adjust(top=0.97, bottom=0.29, left=0.27, right=0.96)
    fig.savefig(relative_folder + "func_val_and_approx_error_" + func_name + ".pdf")

    new_order = [0, 2, 1]
    function_approximation_error_files = [function_approximation_error_files[i] for i in new_order]
    cumulative_msgs_broadcast_disabled_files = [cumulative_msgs_broadcast_disabled_files[i] for i in new_order]
    coordinators = []
    num_messages = []
    max_errors = []

    ############# Figure of automon stats and barchart of -x_1^2 + x_2^2 #############

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='col',
                             gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.35},
                             figsize=get_figsize(wf=0.66, hf=0.8))
    ax0 = axes[0, 0]
    ax1 = axes[1, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 1]

    # Approximation error
    colors = ["tab:blue", "tab:orange", "tab:green"]
    linestyles = ['-', '-', '-']
    for idx, file in enumerate(function_approximation_error_files):
        function_approximation_error = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(function_approximation_error_file_suffix, "")
        if "RLV slack" in coordinator_name:
            coordinator_name = "no ADCD"
        if "RLV no slack" in coordinator_name:
            coordinator_name = "no ADCD no slack"

        if coordinator_name == "no ADCD no slack":
            ax0.plot(np.arange(start_iteration, end_iteration), function_approximation_error[offset:],
                        label=coordinator_name, color=colors[idx], linestyle=linestyles[idx], linewidth=1.1)
        else:
            ax0.plot(np.arange(start_iteration, end_iteration), function_approximation_error[offset:],
                     label=coordinator_name, color=colors[idx], linestyle=linestyles[idx], linewidth=0.8)

        max_errors.append(np.max(function_approximation_error))
        coordinators.append(coordinator_name)

    ax0.set_ylabel("max error")

    # \u03B5 is the unicode of epsilon: r'$\epsilon=$'
    ax0.annotate('\u03B5 = ' + str(error_bound), xy=(0.17, 0.08), xycoords='axes fraction', xytext=(12, 10),
                 textcoords='offset pixels', fontsize=5.2,
                 horizontalalignment='right', verticalalignment='bottom')
    ax0.annotate('', xy=(0.1, 0.06), xycoords='axes fraction', xytext=(0, 10),
                 textcoords='offset pixels', fontsize=5.2,
                 horizontalalignment='right', verticalalignment='bottom', arrowprops=dict(arrowstyle="->", shrinkA=0))

    # Cumulative messages - broadcast disabled
    for idx, file in enumerate(cumulative_msgs_broadcast_disabled_files):
        cumulative_msgs_broadcast_disabled = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(cumulative_msgs_broadcast_disabled_suffix, "")
        if "RLV slack" in coordinator_name:
            coordinator_name = "no ADCD"
        if "RLV no slack" in coordinator_name:
            coordinator_name = "no ADCD no slack"
        ax1.plot(np.arange(start_iteration, end_iteration), cumulative_msgs_broadcast_disabled[offset:],
                    label=coordinator_name, color=colors[idx], linestyle=linestyles[idx], linewidth=0.8)
        assert (coordinator_name == coordinators[idx])
        num_messages.append(cumulative_msgs_broadcast_disabled[-1])

    # Centralization line - every node sends its statistics every second
    ax1.plot(np.arange(start_iteration, end_iteration), np.arange(1, end_iteration - start_iteration + 1) * num_nodes, color="black", linestyle=':', linewidth=0.7)
    ax1.set_ylabel("#messages")
    ax1.set_xlabel("rounds")

    # Error bound (\u03B5 is the unicode of epsilon: r'$\epsilon=$')
    ax0.hlines(error_bound, xmin=start_iteration, xmax=end_iteration - 1, linestyles='dashed', color="black", label='\u03B5 = ' + str(error_bound), linewidth=0.7)

    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax0.spines['bottom'].set_linewidth(0.5)
    ax0.spines['left'].set_linewidth(0.5)
    ax0.tick_params(width=0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.tick_params(width=0.5)

    bar_width = 0.7

    for i, coordinator in enumerate(coordinators):
        ax3.bar([i+0.5], [num_messages[i]], bar_width, label=coordinator, color=colors[i])
    ax3.set_xlim([0, len(coordinators)])

    # horizontal line indicating Centralization
    num_centralization_messages = conf['num_iterations'] * num_nodes - conf['sliding_window_size'] * num_nodes
    ax3.plot([0, len(coordinators)], [num_centralization_messages, num_centralization_messages], ":", linewidth=0.7, color="black")

    ax3.get_xaxis().set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['left'].set_linewidth(0.5)
    ax3.tick_params(width=0.5)

    for i, coordinator in enumerate(coordinators):
        ax2.bar([i+0.5], [max_errors[i]], bar_width, label=coordinator, color=colors[i])

    ax2.get_xaxis().set_visible(False)
    # horizontal line indicating the additive error bound
    ax2.plot([0, len(coordinators)], [error_bound, error_bound], "--", linewidth=0.7, color="black")

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['left'].set_linewidth(0.5)
    ax2.tick_params(width=0.5)

    ax0.set_xlim(left=0)
    ax1.set_xlim(left=0)
    ax1.set_yscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim(ax1.get_ylim())
    ax0.set_ylim(bottom=0)
    ax2.set_ylim(ax0.get_ylim())
    ax1.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax3.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax1.set_yticks([10 ** 1, 10 ** 2, 10 ** 3])
    ax3.set_yticks([10 ** 1, 10 ** 2, 10 ** 3])
    ax1.set_xticks([0, 250, 500, 750, 1000])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    ax0.yaxis.set_label_coords(-0.27, 0.5)
    ax1.yaxis.set_label_coords(-0.27, 0.5)

    ax1.annotate('Centralization', xy=(0.06, 0.6), xycoords='axes fraction', xytext=(35, 12),
                 textcoords='offset pixels', fontsize=5.2,
                 horizontalalignment='right', verticalalignment='bottom')
    ax1.annotate('', xy=(0.08, 0.6), xycoords='axes fraction', xytext=(0, 11),
                 textcoords='offset pixels', fontsize=5.2,
                 horizontalalignment='right', verticalalignment='bottom', arrowprops=dict(arrowstyle="->", shrinkA=0))

    # \u03B5 is the unicode of epsilon: r'$\epsilon=$'
    ax2.annotate('\u03B5 = ' + str(error_bound), xy=(0.19, 0.08), xycoords='axes fraction', xytext=(17, 1),
                 textcoords='offset pixels', fontsize=5.2,
                 horizontalalignment='right', verticalalignment='bottom')

    ax3.annotate('Central.', xy=(0.06, 0.6), xycoords='axes fraction', xytext=(20, 13),
                 textcoords='offset pixels', fontsize=5.2,
                 horizontalalignment='right', verticalalignment='bottom')

    handles, labels = ax1.get_legend_handles_labels()
    plt.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(-1.5, 2.55), columnspacing=1.5, handletextpad=0.6, frameon=False, framealpha=0, handlelength=1.5)

    plt.subplots_adjust(top=0.9, bottom=0.17, left=0.18, right=0.99)
    fig.savefig(relative_folder + "monitoring_stats_" + func_name + ".pdf")

    rcParams.update(rcParamsDefault)


def plot_monitoring_stats_barchart(test_folder, func_name, relative_folder='./'):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    conf = read_config_file(test_folder)
    error_bound = conf["error_bound"]
    num_nodes = conf["num_nodes"]
    num_iterations = conf["num_iterations"]
    offset = get_function_value_offset(test_folder)

    function_approximation_error_file_suffix = "_function_approximation_error.csv"
    function_approximation_error_files = [f for f in os.listdir(test_folder) if f.endswith(function_approximation_error_file_suffix)]
    function_approximation_error_files.sort()
    cumulative_msgs_broadcast_disabled_suffix = "_cumulative_msgs_broadcast_disabled.csv"
    cumulative_msgs_broadcast_disabled_files = [f for f in os.listdir(test_folder) if f.endswith(cumulative_msgs_broadcast_disabled_suffix)]
    cumulative_msgs_broadcast_disabled_files.sort()

    coordinators = []
    num_messages = []
    max_errors = []

    for idx, file in enumerate(cumulative_msgs_broadcast_disabled_files):
        cumulative_msgs_broadcast_disabled = np.genfromtxt(test_folder + "/" + file)
        coordinator_name = file.replace(cumulative_msgs_broadcast_disabled_suffix, "")
        if "RLV slack" in coordinator_name:
            coordinator_name = "no ADCD"
        if "RLV no slack" in coordinator_name:
            coordinator_name = "no ADCD no slack"
        coordinators.append(coordinator_name)
        num_messages.append(cumulative_msgs_broadcast_disabled[-1])

    for idx, file in enumerate(function_approximation_error_files):
        function_approximation_error = np.genfromtxt(test_folder + "/" + file)[offset:]
        coordinator_name = file.replace(function_approximation_error_file_suffix, "")
        if "RLV slack" in coordinator_name:
            coordinator_name = "no ADCD"
        if "RLV no slack" in coordinator_name:
            coordinator_name = "no ADCD no slack"
        max_errors.append(np.max(function_approximation_error))
        assert (coordinator_name == coordinators[idx])

    new_order = [0, 2, 1]
    coordinators = [coordinators[i] for i in new_order]
    num_messages = [num_messages[i] for i in new_order]
    max_errors = [max_errors[i] for i in new_order]

    ############# Figure of automon stats as barchart of MLP-2 #############

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=get_figsize(wf=0.33, hf=1.7618081461187218, b_fixed_height=True))
    ax0 = axes[1]
    ax1 = axes[0]

    bar_width = 0.7
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i, coordinator in enumerate(coordinators):
        ax0.bar([i + 0.5], [num_messages[i]], bar_width, label=coordinator, color=colors[i])
    ax0.set_xlim([0, len(coordinators)])

    # horizontal line indicating Centralization
    num_centralization_messages = num_iterations * num_nodes - conf['sliding_window_size'] * num_nodes
    ax0.plot([0, len(coordinators)], [num_centralization_messages, num_centralization_messages], ":", linewidth=0.7, color="black")

    ax0.set_ylabel("#messages")
    ax0.get_xaxis().set_visible(False)
    ax0.set_yscale('log')
    ax0.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax0.set_yticks([10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4])
    ax0.set_ylim(bottom=1, top=30000)

    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['bottom'].set_linewidth(0.5)
    ax0.spines['left'].set_linewidth(0.5)
    ax0.tick_params(width=0.5)

    for i, coordinator in enumerate(coordinators):
        ax1.bar([i + 0.5], [max_errors[i]], bar_width, label=coordinator, color=colors[i])
    ax1.set_xlim([0, len(coordinators)])

    ax1.set_ylabel("max error")
    ax1.get_xaxis().set_visible(False)
    # horizontal line indicating the additive error bound
    ax1.plot([0, len(coordinators)], [error_bound, error_bound], "--", linewidth=0.7, color="black")
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.tick_params(width=0.5)

    ax0.yaxis.set_label_coords(-0.6, 0.5)
    ax1.yaxis.set_label_coords(-0.6, 0.5)

    # \u03B5 is the unicode of epsilon: r'$\epsilon=$'
    ax1.annotate('\u03B5 = ' + str(error_bound), xy=(0.18, 0.08), xycoords='axes fraction', xytext=(16, 11),
                 textcoords='offset pixels', fontsize=5.2,
                 horizontalalignment='right', verticalalignment='bottom')

    ax0.annotate('Centralization', xy=(0.06, 0.6), xycoords='axes fraction', xytext=(35, 13),
                 textcoords='offset pixels', fontsize=5.2,
                 horizontalalignment='right', verticalalignment='bottom')

    plt.subplots_adjust(top=0.9, bottom=0.17, left=0.5, right=0.99)
    fig.savefig(relative_folder + "monitoring_stats_barchart_" + func_name + ".pdf")


if __name__ == "__main__":
    # Figure 10 (a)
    test_folder = "../test_results/results_ablation_study_quadratic_inverse_2021-07-08_15-40-37"
    plot_monitoring_stats_graph_and_barchart(test_folder, "quadratic_inverse")

    # Figure 10 (b)
    test_folder = "../test_results/results_ablation_study_mlp_2_2021-07-08_15-46-33"
    plot_monitoring_stats_barchart(test_folder, "mlp_2")
