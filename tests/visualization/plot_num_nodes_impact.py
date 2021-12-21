import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from automon_utils.test_utils import read_config_file
import os
import numpy as np
from tests.visualization.utils import get_figsize, reformat_large_tick_values, get_function_value_offset
import matplotlib.ticker as tick


def get_num_nodes_and_num_messages_arr(parent_test_folder):
    num_nodes_folders = list(filter(lambda x: os.path.isdir(os.path.join(parent_test_folder, x)), os.listdir(parent_test_folder)))
    num_nodes_folders.sort(key=lambda num_nodes_folder: int(num_nodes_folder.split('_')[-1]))
    num_nodes_folders = [parent_test_folder + "/" + sub_folder for sub_folder in num_nodes_folders]

    num_nodes_arr = []
    num_messages_arr_automon = []
    num_messages_arr_centralization = []

    for num_nodes_folder in num_nodes_folders:

        conf = read_config_file(num_nodes_folder)
        num_nodes = conf["num_nodes"]
        offset = get_function_value_offset(num_nodes_folder)

        num_nodes_arr.append(num_nodes)

        real_function_value_file_suffix = "_real_function_value.csv"
        real_function_value_files = [f for f in os.listdir(num_nodes_folder) if f.endswith(real_function_value_file_suffix)]
        real_function_value = np.genfromtxt(num_nodes_folder + "/" + real_function_value_files[0])  # Need only one, as they are all the same

        cumulative_msgs_broadcast_disabled_suffix = "_cumulative_msgs_broadcast_disabled.csv"
        cumulative_msgs_broadcast_disabled_files = [f for f in os.listdir(num_nodes_folder) if f.endswith(cumulative_msgs_broadcast_disabled_suffix)]
        cumulative_msgs_broadcast_disabled_files.sort()

        for idx, file in enumerate(cumulative_msgs_broadcast_disabled_files):
            cumulative_msgs_broadcast_disabled = np.genfromtxt(num_nodes_folder + "/" + file)
            coordinator_name = file.replace(cumulative_msgs_broadcast_disabled_suffix, "")
            if coordinator_name == "AutoMon":
                num_messages_arr_automon.append(cumulative_msgs_broadcast_disabled[-1])

        # Centralization point - every node sends its statistics every second and max error is 0
        start_iteration = offset
        data_len = len(real_function_value) - offset
        end_iteration = offset + data_len
        centralization_num_messages = (end_iteration - start_iteration - 1) * num_nodes
        num_messages_arr_centralization.append(centralization_num_messages)

    return num_nodes_arr, num_messages_arr_automon, num_messages_arr_centralization


def plot_num_nodes_impact_on_communication(parent_test_folder):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 6})
    rcParams.update({'font.size': 7})

    num_nodes_arr, num_messages_arr_automon, num_messages_arr_centralization = get_num_nodes_and_num_messages_arr(parent_test_folder)

    fig = plt.figure(figsize=get_figsize(hf=0.4))
    ax = fig.gca()
    ax.set_ylabel('#messages')
    ax.set_xlabel('#nodes')
    ax.plot(num_nodes_arr, num_messages_arr_automon, label="AutoMon", marker="x")
    ax.plot(num_nodes_arr, num_messages_arr_centralization, label="Centralization", color="black", marker='o', markersize=2)
    ax.legend()

    plt.subplots_adjust(top=0.99, bottom=0.26, left=0.17, right=0.99, hspace=0.1, wspace=0.1)
    fig.savefig(parent_test_folder + "/num_nodes_vs_communication.pdf")
    plt.close(fig)

    print("num_messages_arr_automon:", num_messages_arr_automon)
    print("num_messages_arr_centralization:", num_messages_arr_centralization)

    rcParams.update(rcParamsDefault)


def plot_num_nodes_impact_on_communication_combined(parent_test_folder_inner_product, parent_test_folder_mlp_40):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    num_nodes_arr, num_messages_arr_automon_inner_product, num_messages_arr_centralization = get_num_nodes_and_num_messages_arr(parent_test_folder_inner_product)
    num_nodes_arr_, num_messages_arr_automon_mlp_40, num_messages_arr_centralization_ = get_num_nodes_and_num_messages_arr(parent_test_folder_mlp_40)
    assert (np.array_equal(num_nodes_arr, num_nodes_arr_))
    assert (np.array_equal(num_messages_arr_centralization, num_messages_arr_centralization_))

    fig = plt.figure(figsize=get_figsize(wf=0.5, hf=0.7))
    ax = fig.gca()
    ax.set_ylabel('#messages')
    ax.set_xlabel('#nodes')
    ax.plot(num_nodes_arr, num_messages_arr_automon_mlp_40, label=r"MLP-40", marker="x", markersize="4", linestyle="-", color="tab:orange", linewidth=0.8)
    ax.plot(num_nodes_arr, num_messages_arr_automon_inner_product, label=r"Inner Product (d=40)", marker="1", markersize="7", linestyle="--", color="tab:green", linewidth=0.8)
    ax.plot(np.arange(num_nodes_arr[0], num_nodes_arr[-1], 1), np.arange(num_nodes_arr[0], num_nodes_arr[-1], 1)*1000, color="black", linestyle=":", linewidth=0.8)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax.set_yticks([1000, 10000, 100000])
    ax.set_xscale('log')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    fig.legend(framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, columnspacing=1.5, handletextpad=0.29)

    ax.annotate('Centralization',
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-25, 38), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom')
    plt.subplots_adjust(top=0.85, bottom=0.27, left=0.26, right=0.96)
    fig.savefig("num_nodes_vs_communication.pdf")

    plt.close(fig)
    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    # Figure 7 (b): impact of the number of nodes on communication
    parent_test_folder_inner_product = "../test_results/results_test_num_nodes_impact_inner_product_2021-04-04_11-20-41/"
    parent_test_folder_mlp_40 = "../test_results/results_test_num_nodes_impact_mlp_40_2021-04-04_11-30-56/"
    plot_num_nodes_impact_on_communication_combined(parent_test_folder_inner_product, parent_test_folder_mlp_40)
