import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import matplotlib.ticker as tick
from test_utils.stats_analysis_utils import get_period_approximation_error
from test_utils.test_utils import read_config_file
from experiments.visualization.plot_dimensions_stats import get_num_messages
from experiments.visualization.visualization_utils import get_figsize, reformat_large_tick_values


# Create local folder for the results in S3: mkdir test_results/max_error_vs_communication_inner_product_2021-10-05_aws
# cd test_results/max_error_vs_communication_inner_product_2021-10-05_aws
# Download S3 folder with AWS cli, for example: aws s3 cp s3://auto_mon-experiment-results/max_error_vs_comm_inner_product . --recursive


def get_all_test_folders(parent_test_folder):
    folders = []

    sub_folders = list(filter(lambda x: os.path.isdir(os.path.join(parent_test_folder, x)), os.listdir(parent_test_folder)))
    sub_folders = [parent_test_folder + "/" + sub_folder for sub_folder in sub_folders]

    if len(sub_folders) == 0:
        # This is basic test folder
        folders += [parent_test_folder]
    else:
        # This is parent folder. Get its inside basic test folders
        for sub_folder in sub_folders:
            folders += get_all_test_folders(sub_folder)
        folders += [parent_test_folder]

    return folders


def create_approximate_error_file(parent_test_folder):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    test_folders = get_all_test_folders(parent_test_folder)
    test_folders.sort()
    if "inner_product" in parent_test_folder:
        real_function_value = np.genfromtxt("../../datasets/inner_product/real_function_value.csv")
    if "kld" in parent_test_folder:
        real_function_value = np.genfromtxt("../../datasets/air_quality/real_function_value.csv")
    if "quadratic" in parent_test_folder:
        real_function_value = np.genfromtxt("../../datasets/quadratic/real_function_value.csv")
    num_messages_arr_automon = []
    max_error_arr_automon = []
    periods = [1, 2, 3, 4, 5, 10, 15, 20]
    max_error_arr_periodic = []
    num_messages_arr_periodic = []

    for test_folder in test_folders:
        files = [f for f in os.listdir(test_folder) if "nodes" == f]
        if len(files) == 1:
            # Get from one of the nodes (node 0) log folder the file AutoMon_node_x_full_sync_history.csv
            coordinator_sub_folder = test_folder + "/coordinator"
            node_0_sub_folder = test_folder + "/nodes/node_0/"
            full_sync_history = np.genfromtxt(node_0_sub_folder + "AutoMon_node_0_full_sync_history.csv").astype(int)
            full_sync_history = full_sync_history - 1  # data update x in a node is data round x-1 in the coordinator (and entry x-1 in real_function_value)
            x0_values_on_sync = np.take(real_function_value, full_sync_history)
            full_sync_history = np.append(full_sync_history, [len(real_function_value)])
            repeats = full_sync_history[1:] - full_sync_history[:-1]
            approx_func_value = np.repeat(x0_values_on_sync, repeats)
            assert approx_func_value.shape[0] == real_function_value.shape[0]
            approximation_error = np.abs(approx_func_value - real_function_value)
            conf = read_config_file(coordinator_sub_folder)
            num_nodes = conf["num_nodes"]
            num_messages = get_num_messages(coordinator_sub_folder)
            print("max approximation error for error bound", conf["error_bound"], "is:", np.max(approximation_error), "num messages:", num_messages)
            with open(test_folder + "/function_approximation_error.csv", 'wb') as f:
                np.savetxt(f, approximation_error)

            num_messages_arr_automon.append(num_messages)
            max_error_arr_automon.append(np.max(approximation_error))

    print("num_messages_arr_automon=", np.array2string(np.array(num_messages_arr_automon), separator=', '))
    print("max_error_arr_automon=", np.array2string(np.array(max_error_arr_automon), separator=', '))

    for period in periods:
        periodic_approximation_error, periodic_cumulative_msg = get_period_approximation_error(period, real_function_value, num_nodes, 0)
        max_error_arr_periodic.append(np.max(periodic_approximation_error))
        num_messages_arr_periodic.append(periodic_cumulative_msg[-1])

    data_len = len(real_function_value)
    end_iteration = data_len
    centralization_num_messages = (end_iteration + 1) * num_nodes

    # Figure with error as a function of number of messages
    fig, ax = plt.subplots(figsize=get_figsize(hf=0.4))
    ax.set_xlabel('#messages')
    ax.plot(num_messages_arr_automon, max_error_arr_automon, label="AutoMon", marker="x", markersize=3,
            color="tab:blue", linewidth=0.8)
    ax.plot(num_messages_arr_periodic, max_error_arr_periodic, label="Periodic", linestyle="--", marker=".",
            markersize=5, color="tab:green", linewidth=0.8)
    ax.axvline(x=centralization_num_messages, color="black", linestyle=":", label="Centralization", linewidth=0.8)

    # For every x value of AutoMon in the graph (num_messages_arr_automon), show the error bound (error_bound_arr).
    # It shows if AutoMon has violation of the error bound or not.
    # ax.plot(num_messages_arr_automon, error_bound_arr, label=r"$\epsilon$", marker="_", markersize=3, color="black", linewidth=0)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

    rcParams['axes.titlesize'] = 5.8
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    ax.legend()
    plt.subplots_adjust(top=0.99, bottom=0.26, left=0.16, right=0.95, hspace=0.1, wspace=0.1)
    fig.savefig(parent_test_folder + "/max_error_vs_communication.pdf")
    plt.close(fig)

    rcParams.update(rcParamsDefault)


def coordinator_cpu_info(parent_test_folder):
    test_folders = get_all_test_folders(parent_test_folder)
    test_folders.sort()

    for test_folder in test_folders:
        files = [f for f in os.listdir(test_folder) if "coordinator" == f]
        if len(files) == 1:
            coordinator_cpu_info_file = test_folder + "/coordinator/cpu_info.txt"
            with open(coordinator_cpu_info_file, "r") as f:
                cpu_info = f.read()
                cpu_info = cpu_info.split("\n")
                model_name = [line for line in cpu_info if "Model name:" in line][0].split(":")[1].strip()
                print(test_folder, ":", model_name)


def nodes_cpu_info(parent_test_folder):
    test_folders = get_all_test_folders(parent_test_folder)
    test_folders.sort()

    for test_folder in test_folders:
        files = [f for f in os.listdir(test_folder) if "nodes" == f]
        if len(files) == 1:
            nodes_folders = [test_folder + "/nodes/" + f for f in os.listdir(test_folder + "/nodes") if "node_" in f]
            num_nodes = len(nodes_folders)
            print(test_folder)
            for node_idx in range(num_nodes):
                node_cpu_info_file = test_folder + "/nodes/node_" + str(node_idx) + "/cpu_info.txt"
                with open(node_cpu_info_file, "r") as f:
                    cpu_info = f.read()
                    cpu_info = cpu_info.split("\n")
                    model_name = [line for line in cpu_info if "Model name:" in line][0].split(":")[1].strip()
                    print("node_" + str(node_idx), ":", model_name)


def coordinator_num_packets_sent_and_received(parent_test_folder):
    test_folders = get_all_test_folders(parent_test_folder)
    test_folders.sort()

    for test_folder in test_folders:
        files = [f for f in os.listdir(test_folder) if "coordinator" == f]
        if len(files) == 1:
            netstat_info_file = test_folder + "/coordinator/netstat_info.txt"
            with open(netstat_info_file, "r") as f:
                netstat_info = f.read()
                netstat_info = netstat_info.split("\n")
                received_before = [line for line in netstat_info if "eth1" in line][0].split()[2]
                received_after = [line for line in netstat_info if "eth1" in line][1].split()[2]
                sent_before = [line for line in netstat_info if "eth1" in line][0].split()[6]
                sent_after = [line for line in netstat_info if "eth1" in line][1].split()[6]
                print(test_folder, ":", (int(received_after) - int(received_before)) + (int(sent_after) - int(sent_before)))


def nodes_num_packets_sent_and_received(parent_test_folder):
    test_folders = get_all_test_folders(parent_test_folder)
    test_folders.sort()

    for test_folder in test_folders:
        files = [f for f in os.listdir(test_folder) if "nodes" == f]
        if len(files) == 1:
            nodes_folders = [test_folder + "/nodes/" + f for f in os.listdir(test_folder + "/nodes") if "node_" in f]
            num_nodes = len(nodes_folders)
            print(test_folder)
            for node_idx in range(num_nodes):
                netstat_info_file = test_folder + "/nodes/node_" + str(node_idx) + "/netstat_info.txt"
                with open(netstat_info_file, "r") as f:
                    netstat_info = f.read()
                    netstat_info = netstat_info.split("\n")
                    received_before = [line for line in netstat_info if "eth1" in line][0].split()[2]
                    received_after = [line for line in netstat_info if "eth1" in line][1].split()[2]
                    sent_before = [line for line in netstat_info if "eth1" in line][0].split()[6]
                    sent_after = [line for line in netstat_info if "eth1" in line][1].split()[6]
                    print("node_" + str(node_idx), ":", (int(received_after) - int(received_before)) + (int(sent_after) - int(sent_before)))


def ccordinator_nethogs_vs_automon(parent_test_folder):
    test_folders = get_all_test_folders(parent_test_folder)
    test_folders.sort()

    for test_folder in test_folders:
        files = [f for f in os.listdir(test_folder) if "coordinator" == f]
        if len(files) == 1:
            nethogs_vs_automon_file = test_folder + "/coordinator/nethogs_vs_automon.txt"
            with open(nethogs_vs_automon_file, "r") as f:
                info = f.read()
                info = info.split("\n")
                num_received_bytes = int(info[0].split(" ")[1])
                num_sent_bytes = int(info[0].split(" ")[3])
                nethogs_num_received_bytes = int(info[0].split(" ")[5])
                nethogs_num_sent_bytes = int(info[0].split(" ")[7])
                received_diff = nethogs_num_received_bytes - num_received_bytes
                sent_diff = nethogs_num_sent_bytes - num_sent_bytes
                received_diff_percent = received_diff * 100.0 / num_received_bytes
                sent_diff_percent = sent_diff * 100.0 / num_sent_bytes
                print(test_folder, ": received_diff_percent", received_diff_percent, "sent_diff_percent", sent_diff_percent)


if __name__ == "__main__":
    parent_test_folder = "../../examples/test_results/max_error_vs_communication_inner_product_2021-10-05_aws"
    #parent_test_folder = "../../examples/test_results/max_error_vs_communication_quadratic_2021-10-06_aws"
    #parent_test_folder = "../../examples/test_results/max_error_vs_communication_kld_2021-10-07_aws"

    create_approximate_error_file(parent_test_folder)
    #coordinator_cpu_info(parent_test_folder)
    #nodes_cpu_info(parent_test_folder)
    #coordinator_num_packets_sent_and_received(parent_test_folder)
    #nodes_num_packets_sent_and_received(parent_test_folder)
    #ccordinator_nethogs_vs_automon(parent_test_folder)
