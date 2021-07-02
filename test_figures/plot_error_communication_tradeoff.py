import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from test_utils import read_config_file, write_config_to_file
import os
import numpy as np
import matplotlib.ticker as tick
import pickle as pkl
from test_figures.plot_figures_utils import get_figsize, reformat_large_tick_values, get_function_value_offset


def get_period_approximation_error(period, real_function_value, num_nodes, offset):
    stream_len = len(real_function_value) - offset
    f_x0_sync_points = real_function_value[offset::period]
    f_x0_at_every_second = f_x0_sync_points.repeat(period)[:stream_len]
    approximation_error = np.abs(real_function_value[offset:] - f_x0_at_every_second)

    # Step function: at each sync point the step increases by num_nodes, when each node sends its statistics (no sync message needed in periodic)
    cumulative_msgs = np.array([num_nodes * (i + 1) for i in range(len(f_x0_sync_points))])
    cumulative_msgs = cumulative_msgs.repeat(period)[:stream_len]

    approximation_error = np.append(np.zeros(offset), approximation_error)
    cumulative_msgs = np.append(np.zeros(offset), cumulative_msgs)

    return approximation_error, cumulative_msgs


def max_error_vs_communication(parent_test_folder, func_name, ax, b_use_aggregated_data=True):
    parent_test_folder_suffix = parent_test_folder.split('/')[-1]

    if parent_test_folder_suffix not in os.listdir(".") or not b_use_aggregated_data:
        error_bound_folders = list(filter(lambda x: os.path.isdir(os.path.join(parent_test_folder, x)), os.listdir(parent_test_folder)))
        error_bound_folders.sort(key=lambda error_bound_folder: float(error_bound_folder.split('_')[1]))
        error_bound_folders = [parent_test_folder + "/" + sub_folder for sub_folder in error_bound_folders]

        error_bound_arr = []
        period_equiv_arr = []
    
        max_error_arr_automon = []
        avg_error_arr_automon = []
        p99_error_arr_automon = []
        max_error_arr_periodic = []
        max_error_arr_cb = []
    
        num_messages_arr_automon = []
        num_messages_arr_periodic = []
        num_messages_arr_cb = []
    
        for error_bound_folder in error_bound_folders:
    
            conf = read_config_file(error_bound_folder)
            error_bound = conf["error_bound"]
            num_nodes = conf["num_nodes"]
            offset = get_function_value_offset(error_bound_folder)
    
            error_bound_arr.append(error_bound)
    
            real_function_value_file_suffix = "_real_function_value.csv"
            real_function_value_files = [f for f in os.listdir(error_bound_folder) if f.endswith(real_function_value_file_suffix)]
            real_function_value = np.genfromtxt(error_bound_folder + "/" + real_function_value_files[0])  # Need only one, as they are all the same
    
            function_approximation_error_file_suffix = "_function_approximation_error.csv"
            function_approximation_error_files = [f for f in os.listdir(error_bound_folder) if f.endswith(function_approximation_error_file_suffix)]
            function_approximation_error_files.sort()
    
            cumulative_msgs_broadcast_disabled_suffix = "_cumulative_msgs_broadcast_disabled.csv"
            cumulative_msgs_broadcast_disabled_files = [f for f in os.listdir(error_bound_folder) if f.endswith(cumulative_msgs_broadcast_disabled_suffix)]
            cumulative_msgs_broadcast_disabled_files.sort()
    
            for idx, file in enumerate(function_approximation_error_files):
                function_approximation_error = np.genfromtxt(error_bound_folder + "/" + file)[offset:]
                coordinator_name = file.replace(function_approximation_error_file_suffix, "")
                if coordinator_name == "AutoMon":
                    max_error_arr_automon.append(np.max(function_approximation_error))
                    avg_error_arr_automon.append(np.mean(function_approximation_error))
                    p99_error_arr_automon.append(np.percentile(function_approximation_error, 99))
                elif coordinator_name == "CB":
                    max_error_arr_cb.append(np.max(function_approximation_error))
    
            for idx, file in enumerate(cumulative_msgs_broadcast_disabled_files):
                cumulative_msgs_broadcast_disabled = np.genfromtxt(error_bound_folder + "/" + file)
                coordinator_name = file.replace(cumulative_msgs_broadcast_disabled_suffix, "")
                if coordinator_name == "AutoMon":
                    num_messages_arr_automon.append(cumulative_msgs_broadcast_disabled[-1])
                elif coordinator_name == "CB":
                    num_messages_arr_cb.append(cumulative_msgs_broadcast_disabled[-1])
    
        # Periodic num messages and max error
        periods = [1, 2, 3, 4, 5, 10, 15, 20]
        if func_name == "DNN Intrusion Detection":
            periods = list(range(1, 11)) + [15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 200, 500]

        for period_equiv in periods:
            periodic_equiv_approximation_error, periodic_equiv_cumulative_msg = get_period_approximation_error(
                period_equiv, real_function_value, num_nodes, offset)
            max_error_arr_periodic.append(np.max(periodic_equiv_approximation_error))
            num_messages_arr_periodic.append(periodic_equiv_cumulative_msg[-1])
            period_equiv_arr.append(period_equiv)

        if b_use_aggregated_data:
            # Create folder with with aggregated data as pickle files for the figure
            os.mkdir(parent_test_folder_suffix)
            pkl.dump(error_bound_arr, open(parent_test_folder_suffix + "/error_bound_arr_" + func_name + ".pkl", 'wb'))
            pkl.dump(period_equiv_arr, open(parent_test_folder_suffix + "/period_equiv_arr_" + func_name + ".pkl", 'wb'))
            pkl.dump(max_error_arr_automon, open(parent_test_folder_suffix + "/max_error_arr_automon_" + func_name + ".pkl", 'wb'))
            pkl.dump(avg_error_arr_automon, open(parent_test_folder_suffix + "/avg_error_arr_automon_" + func_name + ".pkl", 'wb'))
            pkl.dump(p99_error_arr_automon, open(parent_test_folder_suffix + "/p99_error_arr_automon_" + func_name + ".pkl", 'wb'))
            pkl.dump(max_error_arr_periodic, open(parent_test_folder_suffix + "/max_error_arr_periodic_" + func_name + ".pkl", 'wb'))
            pkl.dump(max_error_arr_cb, open(parent_test_folder_suffix + "/max_error_arr_cb_" + func_name + ".pkl", 'wb'))
            pkl.dump(num_messages_arr_automon, open(parent_test_folder_suffix + "/num_messages_arr_automon_" + func_name + ".pkl", 'wb'))
            pkl.dump(num_messages_arr_periodic, open(parent_test_folder_suffix + "/num_messages_arr_periodic_" + func_name + ".pkl", 'wb'))
            pkl.dump(num_messages_arr_cb, open(parent_test_folder_suffix + "/num_messages_arr_cb_" + func_name + ".pkl", 'wb'))
            write_config_to_file(parent_test_folder_suffix, conf)
            with open(parent_test_folder_suffix + "/real_function_value_" + func_name + ".csv", 'wb') as f:
                np.savetxt(f, real_function_value)

    if b_use_aggregated_data:
        conf = read_config_file(parent_test_folder_suffix)
        num_nodes = conf["num_nodes"]
        offset = get_function_value_offset(parent_test_folder_suffix)
        real_function_value = np.genfromtxt(parent_test_folder_suffix + "/real_function_value_" + func_name + ".csv")
        error_bound_arr = pkl.load(open(parent_test_folder_suffix + "/error_bound_arr_" + func_name + ".pkl", 'rb'))
        period_equiv_arr = pkl.load(open(parent_test_folder_suffix + "/period_equiv_arr_" + func_name + ".pkl", 'rb'))
        max_error_arr_automon = pkl.load(open(parent_test_folder_suffix + "/max_error_arr_automon_" + func_name + ".pkl", 'rb'))
        avg_error_arr_automon = pkl.load(open(parent_test_folder_suffix + "/avg_error_arr_automon_" + func_name + ".pkl", 'rb'))
        p99_error_arr_automon = pkl.load(open(parent_test_folder_suffix + "/p99_error_arr_automon_" + func_name + ".pkl", 'rb'))
        max_error_arr_periodic = pkl.load(open(parent_test_folder_suffix + "/max_error_arr_periodic_" + func_name + ".pkl", 'rb'))
        max_error_arr_cb = pkl.load(open(parent_test_folder_suffix + "/max_error_arr_cb_" + func_name + ".pkl", 'rb'))
        num_messages_arr_automon = pkl.load(open(parent_test_folder_suffix + "/num_messages_arr_automon_" + func_name + ".pkl", 'rb'))
        num_messages_arr_periodic = pkl.load(open(parent_test_folder_suffix + "/num_messages_arr_periodic_" + func_name + ".pkl", 'rb'))
        num_messages_arr_cb = pkl.load(open(parent_test_folder_suffix + "/num_messages_arr_cb_" + func_name + ".pkl", 'rb'))

    # Centralization point - every node sends its statistics every second and max error is 0
    start_iteration = offset
    data_len = len(real_function_value) - offset
    end_iteration = offset + data_len
    centralization_num_messages = (end_iteration - start_iteration + 1) * num_nodes
    if func_name == "DNN Intrusion Detection":
        centralization_num_messages = end_iteration - start_iteration + 1

    num_messages_limit = 13000
    if func_name == "KLD":
        num_messages_limit = 450000
    if func_name == "DNN Intrusion Detection":
        num_messages_limit = 450000
    max_error_arr_cb = [max_error_arr_cb[i] for i in range(len(num_messages_arr_cb)) if num_messages_arr_cb[i] < num_messages_limit]
    num_messages_arr_cb = [m for m in num_messages_arr_cb if m < num_messages_limit]
    max_error_arr_automon = [max_error_arr_automon[i] for i in range(len(num_messages_arr_automon)) if num_messages_arr_automon[i] < num_messages_limit]
    avg_error_arr_automon = [avg_error_arr_automon[i] for i in range(len(num_messages_arr_automon)) if num_messages_arr_automon[i] < num_messages_limit]
    p99_error_arr_automon = [p99_error_arr_automon[i] for i in range(len(num_messages_arr_automon)) if num_messages_arr_automon[i] < num_messages_limit]
    error_bound_arr = [error_bound_arr[i] for i in range(len(num_messages_arr_automon)) if num_messages_arr_automon[i] < num_messages_limit]
    num_messages_arr_automon = [m for m in num_messages_arr_automon if m < num_messages_limit]
    max_error_arr_periodic = [max_error_arr_periodic[i] for i in range(len(num_messages_arr_periodic)) if num_messages_arr_periodic[i] < num_messages_limit]
    num_messages_arr_periodic = [m for m in num_messages_arr_periodic if m < num_messages_limit]

    if func_name == "DNN Intrusion Detection":
        max_error_arr_automon = [max_error_arr_automon[i] for i in range(len(error_bound_arr)) if error_bound_arr[i] <= 0.05]
        avg_error_arr_automon = [avg_error_arr_automon[i] for i in range(len(error_bound_arr)) if error_bound_arr[i] <= 0.05]
        p99_error_arr_automon = [p99_error_arr_automon[i] for i in range(len(error_bound_arr)) if error_bound_arr[i] <= 0.05]
        num_messages_arr_automon = [num_messages_arr_automon[i] for i in range(len(error_bound_arr)) if error_bound_arr[i] <= 0.05]
        error_bound_arr = [m for m in error_bound_arr if m <= 0.05]

    if func_name == "Quadratic":
        max_error_arr_automon = [max_error_arr_automon[i] for i in range(len(error_bound_arr)) if error_bound_arr[i] <= 1.0]
        avg_error_arr_automon = [avg_error_arr_automon[i] for i in range(len(error_bound_arr)) if error_bound_arr[i] <= 1.0]
        p99_error_arr_automon = [p99_error_arr_automon[i] for i in range(len(error_bound_arr)) if error_bound_arr[i] <= 1.0]
        num_messages_arr_automon = [num_messages_arr_automon[i] for i in range(len(error_bound_arr)) if error_bound_arr[i] <= 1.0]
        error_bound_arr = [m for m in error_bound_arr if m <= 1.0]

    ax.set_xlabel('#messages')
    if len(num_messages_arr_cb) > 0:
        ax.plot(num_messages_arr_cb, max_error_arr_cb, label="CB", marker="o", color="tab:red", markersize=4, fillstyle='none', linewidth=0.8)
    ax.plot(num_messages_arr_automon, max_error_arr_automon, label="AutoMon", marker="x", markersize=3, color="tab:blue", linewidth=0.8)
    ax.plot(num_messages_arr_periodic, max_error_arr_periodic, label="Periodic", linestyle="--", marker=".", markersize=5, color="tab:green", linewidth=0.8)
    ax.axvline(x=centralization_num_messages, color="black", linestyle=":", label="Centralization", linewidth=0.8)

    # For every x value of AutoMon in the graph (num_messages_arr_automon), show the error bound (error_bound_arr).
    # It shows if AutoMon has violation of the error bound or not.
    #ax.plot(num_messages_arr_automon, error_bound_arr, label=r"$\epsilon$", marker="_", markersize=3, color="black", linewidth=0)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

    rcParams['axes.titlesize'] = 5.8
    ax.set_title(func_name, pad=3)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    print("error_bound_arr:", error_bound_arr)
    print("period_equiv:", period_equiv_arr)

    print("max_error_arr_automon:", max_error_arr_automon)
    print("num_messages_arr_automon:", num_messages_arr_automon)

    print("max_error_arr_periodic:", max_error_arr_periodic)
    print("num_messages_arr_periodic:", num_messages_arr_periodic)

    return num_messages_arr_automon, max_error_arr_automon, error_bound_arr, avg_error_arr_automon, p99_error_arr_automon, centralization_num_messages


# This function is called only from test_max_error_vs_communication_xxx experiments.
def plot_max_error_vs_communication(parent_test_folder, func_name):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 6})
    rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize=get_figsize(hf=0.4))
    max_error_vs_communication(parent_test_folder, func_name, ax, b_use_aggregated_data=False)
    ax.legend()
    plt.subplots_adjust(top=0.99, bottom=0.26, left=0.16, right=0.95, hspace=0.1, wspace=0.1)
    fig.savefig(parent_test_folder + "/max_error_vs_communication.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_max_error_vs_communication_combined(parent_test_folder_inner_prod, parent_test_folder_quadratic,
                                             parent_test_folder_dnn_intrusion_detection, parent_test_folder_kld):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})


    # Figure with error as a function of number of messages for 4 functions

    # textwidth is 506.295, columnwidth is 241.14749
    fig, axs = plt.subplots(1, 4, figsize=get_figsize(columnwidth=506.295, hf=0.38 * 241.14749 / 506.295))
    axs[0].set_ylabel('max error')

    inner_product_num_messages_arr_automon, inner_product_max_error_arr_automon, inner_product_error_bound_arr, inner_product_avg_error_arr_automon, inner_product_p99_error_arr_automon, inner_product_centralization_num_messages = max_error_vs_communication(parent_test_folder_inner_prod, "Inner Product", axs[0])
    quadratic_num_messages_arr_automon, quadratic_max_error_arr_automon, quadratic_error_bound_arr, quadratic_avg_error_arr_automon, quadratic_p99_error_arr_automon, quadratic_centralization_num_messages = max_error_vs_communication(parent_test_folder_quadratic, "Quadratic", axs[1])
    kld_num_messages_arr_automon, kld_max_error_arr_automon, kld_error_bound_arr, kld_avg_error_arr_automon, kld_p99_error_arr_automon, kld_centralization_num_messages = max_error_vs_communication(parent_test_folder_kld, "KLD", axs[2])
    dnn_num_messages_arr_automon, dnn_max_error_arr_automon, dnn_error_bound_arr, dnn_avg_error_arr_automon, dnn_p99_error_arr_automon, dnn_centralization_num_messages = max_error_vs_communication(parent_test_folder_dnn_intrusion_detection, "DNN Intrusion Detection", axs[3])

    axs[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    axs[1].set_ylim(top=0.2)
    axs[2].set_yticks([0, 0.05, 0.1, 0.15])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, columnspacing=1.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.subplots_adjust(top=0.9, bottom=0.38, left=0.06, right=0.99, wspace=0.35)
    fig.savefig("max_error_vs_communication.pdf")
    plt.close(fig)


    # Figure of % error (relative error with respect to the error bound) only for KLD and DNN intrusion detection

    fig, axs = plt.subplots(1, 2, figsize=get_figsize(columnwidth=241.14749, hf=0.3), sharey=True)
    axs[0].set_ylabel('percent of bound')

    axs[0].set_xlabel("#messages")
    # Vertical line indication centralization
    axs[0].axvline(x=kld_centralization_num_messages, color="black", linestyle=":", label="Centralization", linewidth=0.8)
    axs[0].plot(kld_num_messages_arr_automon, (np.array(kld_max_error_arr_automon) / np.array(kld_error_bound_arr)) * 100, label='max', marker="+", markersize=3, color="C9", linewidth=0.8, linestyle='--')
    axs[0].plot(kld_num_messages_arr_automon, (np.array(kld_p99_error_arr_automon) / np.array(kld_error_bound_arr)) * 100, label='p99', marker="x", markersize=3, color="tab:blue", linewidth=0.8)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['bottom'].set_linewidth(0.5)
    axs[0].spines['left'].set_linewidth(0.5)
    axs[0].tick_params(width=0.5)
    axs[0].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    axs[0].set_title('KLD', pad=3)
    # horizontal line indicating 100%
    #axs[0].plot([0, np.max(kld_num_messages_arr_automon)], [100, 100], "k--", linewidth=0.8)
    axs[0].set_xlim(left=0)

    axs[1].set_xlabel("#messages")
    # Vertical line indication centralization
    axs[1].axvline(x=dnn_centralization_num_messages, color="black", linestyle=":", label="central.", linewidth=0.8)
    axs[1].plot(dnn_num_messages_arr_automon, (np.array(dnn_max_error_arr_automon) / np.array(dnn_error_bound_arr)) * 100, label='max', marker="+", markersize=3, color="C9", linewidth=0.8, linestyle='--')
    axs[1].plot(dnn_num_messages_arr_automon, (np.array(dnn_p99_error_arr_automon) / np.array(dnn_error_bound_arr)) * 100, label='p99', marker="x", markersize=3, color='tab:blue', linewidth=0.8)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['bottom'].set_linewidth(0.5)
    axs[1].spines['left'].set_linewidth(0.5)
    axs[1].tick_params(width=0.5)
    axs[1].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    axs[1].set_title('DNN Intrusion Detection', pad=3)
    # horizontal line indicating 100%
    #axs[1].plot([0, np.max(dnn_num_messages_arr_automon)], [100, 100], "k--", linewidth=0.8)
    axs[1].set_xlim(left=0)

    axs[0].set_yticks([0, 50, 100, 150, 200])
    axs[1].set_yticks([0, 50, 100, 150, 200])
    axs[0].set_ylim(top=120)
    axs[1].set_ylim(top=120)
    axs[0].yaxis.set_major_formatter(tick.PercentFormatter())
    axs[0].set_xticks([0, 100000, 200000, 300000, 400000])
    axs[1].set_xticks([0, 100000, 200000, 300000, 400000])

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=1, frameon=False, handletextpad=0.3, handlelength=1.85, bbox_to_anchor=(1.02, 1))

    plt.subplots_adjust(top=0.89, bottom=0.31, left=0.14, right=0.85, wspace=0.2)
    fig.savefig("percent_error_kld_and_dnn.pdf")
    plt.close(fig)

    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    # Figure 5 in the paper

    parent_test_folder_prefix = "../test_results/results_test_max_error_vs_communication_"
    parent_test_folder_inner_prod = parent_test_folder_prefix + "inner_product_2021-04-04_06-37-07"
    parent_test_folder_quadratic = parent_test_folder_prefix + "quadratic_2021-04-13_15-32-27"
    parent_test_folder_dnn_intrusion_detection = parent_test_folder_prefix + "dnn_intrusion_detection_2021-04-10_14-13-40"
    parent_test_folder_kld = parent_test_folder_prefix + "kld_air_quality_2021-07-01_14-11-49"

    # Remote
    plot_max_error_vs_communication_combined(parent_test_folder_inner_prod, parent_test_folder_quadratic,
                                             parent_test_folder_dnn_intrusion_detection, parent_test_folder_kld)
