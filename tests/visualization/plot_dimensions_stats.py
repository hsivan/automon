import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import numpy as np
from tests.visualization.utils import get_figsize, reformat_large_tick_values
from utils.test_utils import read_config_file
import os
import matplotlib.ticker as tick


def get_mean_and_std(test_folder, line_str):
    conf = read_config_file(test_folder)
    num_nodes = conf["num_nodes"]
    log_file = [f for f in os.listdir(test_folder) if f.endswith(".log")][0]
    log_file = open(test_folder + "/" + log_file, 'r')
    log = log_file.read()
    log_lines = log.split("\n")

    line_splitter = line_str.split(" ")[-1] + " "
    avg_time_seconds = [float(line.split(line_splitter)[-1]) for line in log_lines if "Avg " + line_str in line]
    std_time_seconds = [float(line.split(line_splitter)[-1]) for line in log_lines if "Std " + line_str in line]
    if len(avg_time_seconds) == 0:
        avg_time_seconds = 0.0
        std_time_seconds = 0.0
    else:
        full_line = [line for line in log_lines if "Avg " + line_str in line][0]
        if "Node" in full_line:
            # Take num_nodes last numbers and get their avg. The other numbers in the list are from the neighborhood tuning.
            avg_time_seconds = np.mean(avg_time_seconds[-num_nodes:])
            std_time_seconds = np.mean(std_time_seconds[-num_nodes:])
        else:
            # This is the coordinator. Take only the last number. The other numbers in the list are from the neighborhood tuning.
            avg_time_seconds = avg_time_seconds[-1]
            std_time_seconds = std_time_seconds[-1]
    return avg_time_seconds, std_time_seconds


def get_run_time(test_folder):
    log_file = [f for f in os.listdir(test_folder) if f.endswith(".log")][0]
    log_file = open(test_folder + "/" + log_file, 'r')
    log = log_file.read()
    log = log.split("\n")
    run_time_seconds = float([line for line in log if "The test took" in line][-1].split(": ")[1].split(" seconds")[0])
    return run_time_seconds


def get_num_messages(test_folder):
    results_file = open(test_folder + "/results.txt", 'r')
    results = results_file.read()
    results = results.split("\n")
    num_messages = int([result for result in results if "Total msgs broadcast enabled" in result][-1].split("disabled ")[1])
    return num_messages


def get_dim_and_num_messages(test_folder):
    conf = read_config_file(test_folder)
    dim = conf["d"]
    num_messages = get_num_messages(test_folder)
    return dim, num_messages


def get_centralization_total_num_messages(test_folder):
    dimension_folders = list(filter(lambda x: os.path.isdir(os.path.join(test_folder, x)), os.listdir(test_folder)))
    dimension_folders.sort()
    dimension_folders = [test_folder + "/" + sub_folder for sub_folder in dimension_folders]
    conf = read_config_file(dimension_folders[0])
    centralization_num_messages = conf['num_iterations'] * conf['num_nodes'] - conf['sliding_window_size'] * conf['num_nodes']
    return centralization_num_messages


def get_time_stats_arrs(test_folder):
    dimension_arr = []
    num_messages_arr = []
    node_avg_data_update_time_arr = []
    node_std_data_update_time_arr = []
    coordinator_avg_full_sync_time_arr = []
    coordinator_std_full_sync_time_arr = []
    node_avg_inside_safe_zone_evaluation_arr = []
    node_std_inside_safe_zone_evaluation_arr = []
    node_avg_inside_bounds_evaluation_arr = []
    node_std_inside_bounds_evaluation_arr = []

    dimension_folders = list(filter(lambda x: os.path.isdir(os.path.join(test_folder, x)), os.listdir(test_folder)))
    dimension_folders.sort(key=lambda dimension_folder: int(dimension_folder.split('_')[1]))
    dimension_folders = [test_folder + "/" + sub_folder for sub_folder in dimension_folders]

    for idx, dimension_folder in enumerate(dimension_folders):
        dim, num_messages = get_dim_and_num_messages(dimension_folder)
        coordinator_avg_full_sync, coordinator_std_full_sync = get_mean_and_std(dimension_folder, "full sync time (ignore first time)")
        node_avg_inside_safe_zone_evaluation, node_std_inside_safe_zone_evaluation = get_mean_and_std(dimension_folder, "inside_safe_zone evaluation time")
        node_avg_inside_bounds_evaluation, node_std_inside_bound_evaluation = get_mean_and_std(dimension_folder, "inside_bounds evaluation time")
        node_avg_data_update, node_std_data_update = get_mean_and_std(dimension_folder, "data update time")
        print(dim, num_messages)
        dimension_arr.append(dim)
        num_messages_arr.append(num_messages)
        node_avg_data_update_time_arr.append(node_avg_data_update)
        node_std_data_update_time_arr.append(node_std_data_update)
        coordinator_avg_full_sync_time_arr.append(coordinator_avg_full_sync)
        coordinator_std_full_sync_time_arr.append(coordinator_std_full_sync)
        node_avg_inside_safe_zone_evaluation_arr.append(node_avg_inside_safe_zone_evaluation)
        node_std_inside_safe_zone_evaluation_arr.append(node_std_inside_safe_zone_evaluation)
        node_avg_inside_bounds_evaluation_arr.append(node_avg_inside_bounds_evaluation)
        node_std_inside_bounds_evaluation_arr.append(node_std_inside_bound_evaluation)

    sec_to_ms = 1000
    return np.array(dimension_arr), np.array(num_messages_arr),\
           np.array(node_avg_data_update_time_arr)*sec_to_ms, np.array(node_std_data_update_time_arr)*sec_to_ms,\
           np.array(coordinator_avg_full_sync_time_arr)*sec_to_ms, np.array(coordinator_std_full_sync_time_arr)*sec_to_ms,\
           np.array(node_avg_inside_safe_zone_evaluation_arr)*sec_to_ms, np.array(node_std_inside_safe_zone_evaluation_arr)*sec_to_ms,\
           np.array(node_avg_inside_bounds_evaluation_arr)*sec_to_ms, np.array(node_std_inside_bounds_evaluation_arr)*sec_to_ms


def plot_dimensions_figures(kld_test_folder, inner_product_test_folder, mlp_test_folder):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    dimension_arr_kld, num_messages_arr_kld, node_avg_data_update_time_arr_kld, node_std_data_update_time_arr_kld, \
    coordinator_avg_full_sync_time_arr_kld, coordinator_std_full_sync_time_arr_kld, \
    node_avg_inside_safe_zone_evaluation_arr_kld, node_std_inside_safe_zone_evaluation_arr_kld, \
    node_avg_inside_bounds_evaluation_arr_kld, node_std_inside_bounds_evaluation_arr_kld = get_time_stats_arrs(kld_test_folder)

    dimension_arr_inner_prod, num_messages_arr_inner_prod, node_avg_data_update_time_arr_inner_prod, node_std_data_update_time_arr_inner_prod, \
    coordinator_avg_full_sync_time_arr_inner_prod, coordinator_std_full_sync_time_arr_inner_prod, \
    node_avg_inside_safe_zone_evaluation_arr_inner_prod, node_std_inside_safe_zone_evaluation_arr_inner_prod, \
    node_avg_inside_bounds_evaluation_arr_inner_prod, node_std_inside_bounds_evaluation_arr_inner_prod = get_time_stats_arrs(inner_product_test_folder)

    dimension_arr_mlp, num_messages_arr_mlp, node_avg_data_update_time_arr_mlp, node_std_data_update_time_arr_mlp, \
    coordinator_avg_full_sync_time_arr_mlp, coordinator_std_full_sync_time_arr_mlp, \
    node_avg_inside_safe_zone_evaluation_arr_mlp, node_std_inside_safe_zone_evaluation_arr_mlp, \
    node_avg_inside_bounds_evaluation_arr_mlp, node_std_inside_bounds_evaluation_arr_mlp = get_time_stats_arrs(mlp_test_folder)

    assert (np.all(dimension_arr_kld == dimension_arr_inner_prod))
    assert (np.all(dimension_arr_kld == dimension_arr_mlp))
    dimension_arr = dimension_arr_kld

    # Total number of messages is every node sends it local vector in every round
    centralization_num_messages_kld = get_centralization_total_num_messages(kld_test_folder)
    centralization_num_messages_inner_prod = get_centralization_total_num_messages(inner_product_test_folder)
    centralization_num_messages_mlp = get_centralization_total_num_messages(mlp_test_folder)
    assert (centralization_num_messages_kld == centralization_num_messages_inner_prod)
    assert (centralization_num_messages_kld == centralization_num_messages_mlp)
    centralization_num_messages = centralization_num_messages_kld


    ################# Communication figure #################
    fig = plt.figure(figsize=get_figsize(wf=0.5, hf=0.7))
    ax = fig.gca()
    ax.set_ylabel('#messages')
    ax.set_xlabel('dimension')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.set_ylim(bottom=0, top=13000)
    ax.set_yticks([0, 5000, 10000])
    ax.set_xticks([10, 50, 100, 150, 200])
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax.hlines(centralization_num_messages, xmin=dimension_arr[0], xmax=dimension_arr[-1], linestyles=':', linewidth=1, color="black")
    ax.plot(dimension_arr, num_messages_arr_kld, linestyle='-', marker='.', markersize=5, linewidth=1, label="KLD")
    ax.plot(dimension_arr, num_messages_arr_mlp, linestyle='--', marker='x', markersize=4, linewidth=1, label="MLP-d")
    ax.plot(dimension_arr, num_messages_arr_inner_prod, linestyle='-.', marker="1", markersize=5, linewidth=1, label="Inner Product")
    fig.legend(framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, columnspacing=0.8, handletextpad=0.29)
    # use negative points or pixels to specify from right, top -10, 10
    # is 10 points to the left of the right side of the axes and 10
    # points above the bottom
    ax.annotate('Centralization',
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-44, 38), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom')
    plt.subplots_adjust(top=0.85, bottom=0.27, left=0.23, right=0.95)
    fig.savefig("dimension_communication.pdf")


    ################# Node runtime on data update figure #################
    fig = plt.figure(figsize=get_figsize(wf=0.5, hf=0.7))
    ax = fig.gca()
    ax.set_ylabel('update time (ms)')
    ax.set_xlabel('dimension')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.plot(dimension_arr, node_avg_data_update_time_arr_kld, linestyle='-', marker='.', markersize=5, linewidth=1,
            label="KLD")
    ax.fill_between(dimension_arr,
                    node_avg_data_update_time_arr_kld - node_std_data_update_time_arr_kld,
                    node_avg_data_update_time_arr_kld + node_std_data_update_time_arr_kld, alpha=.3)
    ax.plot(dimension_arr, node_avg_data_update_time_arr_mlp, linestyle='--', marker='x', markersize=4, linewidth=1,
            label="MLP-d")
    ax.fill_between(dimension_arr,
                    node_avg_data_update_time_arr_mlp - node_std_data_update_time_arr_mlp,
                    node_avg_data_update_time_arr_mlp + node_std_data_update_time_arr_mlp, alpha=.3)
    ax.plot(dimension_arr, node_avg_data_update_time_arr_inner_prod, linestyle='-.', marker="1", markersize=5,
            linewidth=1, label="Inner Product")
    ax.fill_between(dimension_arr,
                    node_avg_data_update_time_arr_inner_prod - node_std_data_update_time_arr_inner_prod,
                    node_avg_data_update_time_arr_inner_prod + node_std_data_update_time_arr_inner_prod, alpha=.3)
    ax.set_yscale('log')
    ax.set_yticks([10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1])
    ax.set_xticks([10, 50, 100, 150, 200])
    plt.subplots_adjust(top=0.85, bottom=0.27, left=0.28, right=0.99)
    fig.savefig("dimension_node_runtime.pdf")


    ################# Coordinator full sync runtime figure #################
    fig = plt.figure(figsize=get_figsize(wf=0.5, hf=0.7))
    ax = fig.gca()
    ax.set_ylabel('full sync time (ms)')
    ax.set_xlabel('dimension')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.plot(dimension_arr, coordinator_avg_full_sync_time_arr_kld, linestyle='-', marker='.', markersize=5, linewidth=1, label="KLD")
    ax.fill_between(dimension_arr,
                    coordinator_avg_full_sync_time_arr_kld - coordinator_std_full_sync_time_arr_kld,
                    coordinator_avg_full_sync_time_arr_kld + coordinator_std_full_sync_time_arr_kld, alpha=.3)
    print("coordinator_avg_full_sync_time_arr_kld", coordinator_avg_full_sync_time_arr_kld)
    ax.plot(dimension_arr, coordinator_avg_full_sync_time_arr_mlp, linestyle='--', marker='x', markersize=4, linewidth=1, label="MLP-d")
    ax.fill_between(dimension_arr,
                    coordinator_avg_full_sync_time_arr_mlp - coordinator_std_full_sync_time_arr_mlp,
                    coordinator_avg_full_sync_time_arr_mlp + coordinator_std_full_sync_time_arr_mlp, alpha=.3)
    ax.plot(dimension_arr, coordinator_avg_full_sync_time_arr_inner_prod, linestyle='-.', marker="1", markersize=5, linewidth=1, label="Inner Product")
    ax.fill_between(dimension_arr,
                    coordinator_avg_full_sync_time_arr_inner_prod - coordinator_std_full_sync_time_arr_inner_prod,
                    coordinator_avg_full_sync_time_arr_inner_prod + coordinator_std_full_sync_time_arr_inner_prod, alpha=.3)
    ax.set_yscale('log')
    ax.set_xticks([10, 50, 100, 150, 200])
    ax.set_yticks([10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])
    fig.legend(framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, columnspacing=0.8,
               handletextpad=0.29)
    plt.subplots_adjust(top=0.85, bottom=0.27, left=0.23, right=0.95)
    fig.savefig("dimension_coordinator_runtime.pdf")


    ################# Node runtime on inside_bounds and inside_safe_zone figure #################
    fig = plt.figure(figsize=get_figsize(hf=0.4))
    ax = fig.gca()
    ax.set_ylabel('node runtime (ms)')
    ax.set_xlabel('dimension')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    bar_width = 0.15
    chosen_dimensions_indices = np.where(np.isin(dimension_arr, [10, 100, 200]))
    chosen_dimensions = dimension_arr[chosen_dimensions_indices]
    index = np.arange(len(chosen_dimensions))
    error_kw = dict(elinewidth=0.5, capsize=0.8, capthick=0.01)
    rcParams['hatch.linewidth'] = 0.3
    ax.bar(index + 0 * bar_width, node_avg_inside_safe_zone_evaluation_arr_kld[chosen_dimensions_indices], bar_width, yerr=node_std_inside_safe_zone_evaluation_arr_kld[chosen_dimensions_indices], label="KLD, check safe zone", color='tab:blue', error_kw=error_kw, edgecolor='black', linewidth=0.2)
    ax.bar(index + 2 * bar_width, node_avg_inside_safe_zone_evaluation_arr_mlp[chosen_dimensions_indices], bar_width, yerr=node_std_inside_safe_zone_evaluation_arr_mlp[chosen_dimensions_indices], label="MLP-d, check safe zone", color='tab:orange', edgecolor='black', linewidth=0.2, error_kw=error_kw)
    ax.bar(index + 4 * bar_width, node_avg_inside_safe_zone_evaluation_arr_inner_prod[chosen_dimensions_indices], bar_width, yerr=node_std_inside_safe_zone_evaluation_arr_inner_prod[chosen_dimensions_indices], label="Inner Product, check safe zone", color='tab:green', edgecolor='black', linewidth=0.2, error_kw=error_kw)
    ax.bar(index + 1 * bar_width, node_avg_inside_bounds_evaluation_arr_kld[chosen_dimensions_indices], bar_width, yerr=node_std_inside_bounds_evaluation_arr_kld[chosen_dimensions_indices], label=r"KLD, compute f", color='lightblue', error_kw=error_kw, edgecolor='black', linewidth=0.2, hatch="xxxx")
    ax.bar(index + 3 * bar_width, node_avg_inside_bounds_evaluation_arr_mlp[chosen_dimensions_indices], bar_width, yerr=node_std_inside_bounds_evaluation_arr_mlp[chosen_dimensions_indices], label=r"MLP-d, compute f", color='orange', error_kw=error_kw, edgecolor='black', linewidth=0.2, hatch="xxxx")
    ax.bar(index + 5 * bar_width, node_avg_inside_bounds_evaluation_arr_inner_prod[chosen_dimensions_indices], bar_width, yerr=node_std_inside_bounds_evaluation_arr_inner_prod[chosen_dimensions_indices], label=r"Inner Product, compute f", color='lightgreen', error_kw=error_kw, edgecolor='black', linewidth=0.2, hatch="xxxx")
    plt.xticks(index + 2.5 * bar_width, [str(dim) for dim in chosen_dimensions])

    ax.set_yscale('log')
    ax.set_yticks([10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0])
    plt.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.6), frameon=False, labelspacing=0.5)
    plt.subplots_adjust(top=0.75, bottom=0.25, left=0.14, right=0.99)
    fig.savefig("dimension_node_runtime_in_parts.pdf")

    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    # Figure 7 (a) and more (scalability to dimensions and runtime)
    kld_test_folder = "../test_results/results_test_dimension_impact_kld_air_quality_2021-10-10_16-00-18"
    inner_product_test_folder = "../test_results/results_test_dimension_impact_inner_product_2021-10-10_13-31-18"
    mlp_test_folder = "../test_results/results_test_dimension_impact_mlp_2021-10-10_13-40-24"
    plot_dimensions_figures(kld_test_folder, inner_product_test_folder, mlp_test_folder)
