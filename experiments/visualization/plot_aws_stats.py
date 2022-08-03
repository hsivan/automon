import os
import sys
import struct
import numpy as np
import pickle
from automon.common_messages import messages_header_format
from test_utils.stats_analysis_utils import get_period_approximation_error
from test_utils.test_utils import read_config_file
from experiments.visualization.plot_dimensions_stats import get_num_messages
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import matplotlib.ticker as tick
import datetime
from experiments.visualization.visualization_utils import get_figsize, reformat_large_tick_values

# Create local folder for the results in S3: mkdir test_results/max_error_vs_communication_inner_product_aws_2021-10-05
# cd test_results/max_error_vs_communication_inner_product_aws_2021-10-05
# Download S3 folder with AWS cli, for example: aws s3 cp s3://automon-experiment-results/max_error_vs_comm_inner_product . --recursive


def get_num_bytes(test_folder):
    results_file = open(test_folder + "/results.txt", 'r')
    results = results_file.read()
    results = results.split("\n")
    bytes_sent = int([result for result in results if "Bytes sent" in result][-1].split("sent ")[1])
    bytes_received = int([result for result in results if "Bytes received" in result][-1].split("received ")[1])
    num_bytes = bytes_sent + bytes_received
    return num_bytes


def get_periodic_message_size(test_folder):
    conf = read_config_file(test_folder)
    x_len = conf["d"]
    messages_payload_format = struct.Struct('! %dd' % x_len)  # local_vector
    return messages_payload_format.size + messages_header_format.size


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


def get_num_messages_and_max_error_sim(parent_test_folder):
    error_bound_folders = list(filter(lambda x: os.path.isdir(os.path.join(parent_test_folder, x)), os.listdir(parent_test_folder)))
    error_bound_folders.sort(key=lambda error_bound_folder: float(error_bound_folder.split('_')[1]))
    error_bound_folders = [parent_test_folder + "/" + sub_folder for sub_folder in error_bound_folders]

    error_bound_arr = []
    max_error_arr_automon = []
    num_messages_arr_automon = []

    for error_bound_folder in error_bound_folders:
        conf = read_config_file(error_bound_folder)
        error_bound = conf["error_bound"]
        num_nodes = conf["num_nodes"]

        error_bound_arr.append(error_bound)

        real_function_value_file_suffix = "_real_function_value.csv"
        real_function_value_files = [f for f in os.listdir(error_bound_folder) if f.endswith(real_function_value_file_suffix)]
        real_function_value = np.genfromtxt(error_bound_folder + "/" + real_function_value_files[0])  # Need only one, as they are all the same

        function_approximation_error_file_suffix = "_function_approximation_error.csv"
        function_approximation_error_files = [f for f in os.listdir(error_bound_folder) if f.endswith(function_approximation_error_file_suffix)]
        function_approximation_error_files.sort()

        cumulative_msgs_broadcast_disabled_suffix = "_cumulative_msgs_broadcast_disabled.csv"
        cumulative_msgs_broadcast_disabled_files = [f for f in os.listdir(error_bound_folder) if
                                                    f.endswith(cumulative_msgs_broadcast_disabled_suffix)]
        cumulative_msgs_broadcast_disabled_files.sort()

        for idx, file in enumerate(function_approximation_error_files):
            function_approximation_error = np.genfromtxt(error_bound_folder + "/" + file)
            coordinator_name = file.replace(function_approximation_error_file_suffix, "")
            if coordinator_name == "AutoMon":
                max_error_arr_automon.append(np.max(function_approximation_error))

        for idx, file in enumerate(cumulative_msgs_broadcast_disabled_files):
            cumulative_msgs_broadcast_disabled = np.genfromtxt(error_bound_folder + "/" + file)
            coordinator_name = file.replace(cumulative_msgs_broadcast_disabled_suffix, "")
            if coordinator_name == "AutoMon":
                num_messages_arr_automon.append(cumulative_msgs_broadcast_disabled[-1])

    return error_bound_arr, num_messages_arr_automon, max_error_arr_automon


def ccordinator_nethogs_vs_automon(parent_test_folder):
    test_folders = get_all_test_folders(parent_test_folder)
    test_folders.sort()
    num_bytes_arr_zmq = []

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
                num_bytes_arr_zmq.append(nethogs_num_received_bytes + nethogs_num_sent_bytes)
    return num_bytes_arr_zmq


def get_num_messages_and_max_error_aws(parent_test_folder):
    test_folders = get_all_test_folders(parent_test_folder)
    test_folders.sort()
    if "inner_product" in parent_test_folder:
        data_folder = relative_path + 'datasets/inner_product/'
        real_function_value = np.genfromtxt(data_folder + "real_function_value.csv")
    if "kld" in parent_test_folder:
        data_folder = relative_path + 'datasets/air_quality/'
        real_function_value = np.genfromtxt(data_folder + "real_function_value.csv")
    if "quadratic" in parent_test_folder:
        data_folder = relative_path + 'datasets/quadratic/'
        real_function_value = np.genfromtxt(data_folder + "real_function_value.csv")
    if "dnn" in parent_test_folder:
        data_folder = relative_path + 'datasets/intrusion_detection/'
        real_function_value = np.genfromtxt(data_folder + "real_function_value.csv")

    error_bound_arr = []
    num_messages_arr_automon = []
    max_error_arr_automon = []
    num_bytes_arr_automon = []
    num_bytes_arr_zmq = ccordinator_nethogs_vs_automon(parent_test_folder)

    for test_folder in test_folders:
        files = [f for f in os.listdir(test_folder) if "nodes" == f]
        if len(files) == 1:
            # Get from one of the nodes (node 0) log folder the file AutoMon_node_x_full_sync_history.csv
            coordinator_sub_folder = test_folder + "/coordinator"
            node_0_sub_folder = test_folder + "/nodes/node_0/"
            if "dnn" in parent_test_folder:
                # For DNN the full_sync_history comes from node 0 (or any other node) log: detected State.WaitForSync after XXX data updates.
                # The reason to use the log is that the node doesn't know the global number of data rounds (for the other functions
                # the data rounds are the same as the node updates).
                full_sync_history = [1]
                for node_idx in range(9):
                    try:
                        node_sub_folder = test_folder + "/nodes/node_" + str(node_idx) + "/"
                        with open(node_sub_folder + "distributed_dnn_intrusion_detection.log", 'r') as f:
                            log = f.read().split("\n")
                            log = [line for line in log if "detected State.WaitForSync" in line]
                            for line in log:
                                full_sync_history.append(int(line.split("after ")[1].split(" data")[0]))
                    except:
                        print("Could not find node", node_idx, "logs for", test_folder)
                full_sync_history.sort()
                full_sync_history = np.unique(full_sync_history)
            else:
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
            error_bound = conf["error_bound"]
            periodic_message_size = get_periodic_message_size(coordinator_sub_folder)
            error_bound_arr.append(error_bound)
            num_messages = get_num_messages(coordinator_sub_folder)
            num_bytes = get_num_bytes(coordinator_sub_folder)
            num_bytes_arr_automon.append(num_bytes)
            with open(test_folder + "/function_approximation_error.csv", 'wb') as f:
                np.savetxt(f, approximation_error)

            num_messages_arr_automon.append(num_messages)
            max_error_arr_automon.append(np.max(approximation_error))

    return error_bound_arr, num_messages_arr_automon, max_error_arr_automon, real_function_value, num_nodes, num_bytes_arr_automon, periodic_message_size, num_bytes_arr_zmq


# Transfer volume in MB (centralization, which is data size with message header that is composed of message type, node idx, and payload len, with network overhead)
def get_centralization_transfer_volume(parent_test_folder_centralization):
    with open(parent_test_folder_centralization + "/coordinator/nethogs.txt", 'r') as f:
        log = f.read()
        num_received_bytes = int(log.split("nethogs_num_received_bytes ")[1].split(" ")[0])
        num_sent_bytes = int(log.split("nethogs_num_sent_bytes ")[1].split(" ")[0])
        transfer_volume = num_received_bytes + num_sent_bytes
    return transfer_volume


def read_data(parent_test_folder_sim, parent_test_folder_aws, parent_test_folder_centralization, function, b_print=False):
    aws_result_folder = "./aws_results/" + function + "/"
    if not os.path.isdir(aws_result_folder):
        max_error_arr_periodic = []
        num_messages_arr_periodic = []
        num_bytes_arr_periodic = []

        sim_error_bound_arr, sim_num_messages_arr_automon, sim_max_error_arr_automon = get_num_messages_and_max_error_sim(parent_test_folder_sim)
        aws_error_bound_arr, aws_num_messages_arr_automon, aws_max_error_arr_automon, real_function_value, num_nodes, aws_num_bytes_arr_automon, periodic_message_size, aws_num_bytes_arr_zmq = get_num_messages_and_max_error_aws(parent_test_folder_aws)

        periods = [1, 2, 3, 4, 5, 10, 15, 20]
        if "dnn" in parent_test_folder_sim:
            periods = list(range(6, 11)) + [15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 200, 500]
        for period in periods:
            periodic_approximation_error, periodic_cumulative_msg = get_period_approximation_error(period, real_function_value, num_nodes, 0)
            max_error_arr_periodic.append(np.max(periodic_approximation_error))
            num_messages_periodic = periodic_cumulative_msg[-1]
            num_messages_arr_periodic.append(num_messages_periodic)
            num_bytes_periodic = num_messages_periodic * periodic_message_size
            num_bytes_arr_periodic.append(num_bytes_periodic)
        centralization_num_messages = (len(real_function_value) + 1) * num_nodes
        if "dnn" in parent_test_folder_sim:
            centralization_num_messages = len(real_function_value) + 1

        centralization_transfer_volume = get_centralization_transfer_volume(parent_test_folder_centralization)

        os.makedirs(aws_result_folder)
        pickle.dump(aws_error_bound_arr, open(aws_result_folder + "aws_error_bound_arr.pkl", 'wb'))
        pickle.dump(aws_num_messages_arr_automon, open(aws_result_folder + "aws_num_messages_arr_automon.pkl", 'wb'))
        pickle.dump(aws_max_error_arr_automon, open(aws_result_folder + "aws_max_error_arr_automon.pkl", 'wb'))
        pickle.dump(real_function_value, open(aws_result_folder + "real_function_value.pkl", 'wb'))
        pickle.dump(num_nodes, open(aws_result_folder + "num_nodes.pkl", 'wb'))
        pickle.dump(aws_num_bytes_arr_automon, open(aws_result_folder + "aws_num_bytes_arr_automon.pkl", 'wb'))
        pickle.dump(periodic_message_size, open(aws_result_folder + "periodic_message_size.pkl", 'wb'))
        pickle.dump(num_messages_arr_periodic, open(aws_result_folder + "num_messages_arr_periodic.pkl", 'wb'))
        pickle.dump(max_error_arr_periodic, open(aws_result_folder + "max_error_arr_periodic.pkl", 'wb'))
        pickle.dump(aws_num_bytes_arr_zmq, open(aws_result_folder + "aws_num_bytes_arr_zmq.pkl", 'wb'))
        pickle.dump(centralization_transfer_volume, open(aws_result_folder + "centralization_transfer_volume.pkl", 'wb'))
        pickle.dump(num_bytes_arr_periodic, open(aws_result_folder + "num_bytes_arr_periodic.pkl", 'wb'))
        pickle.dump(centralization_num_messages, open(aws_result_folder + "centralization_num_messages.pkl", 'wb'))
        pickle.dump(sim_error_bound_arr, open(aws_result_folder + "sim_error_bound_arr.pkl", 'wb'))
        pickle.dump(sim_num_messages_arr_automon, open(aws_result_folder + "sim_num_messages_arr_automon.pkl", 'wb'))
        pickle.dump(sim_max_error_arr_automon, open(aws_result_folder + "sim_max_error_arr_automon.pkl", 'wb'))
    else:
        aws_error_bound_arr = pickle.load(open(aws_result_folder + "aws_error_bound_arr.pkl", 'rb'))
        aws_num_messages_arr_automon = pickle.load(open(aws_result_folder + "aws_num_messages_arr_automon.pkl", 'rb'))
        aws_max_error_arr_automon = pickle.load(open(aws_result_folder + "aws_max_error_arr_automon.pkl", 'rb'))
        real_function_value = pickle.load(open(aws_result_folder + "real_function_value.pkl", 'rb'))
        num_nodes = pickle.load(open(aws_result_folder + "num_nodes.pkl", 'rb'))
        aws_num_bytes_arr_automon = pickle.load(open(aws_result_folder + "aws_num_bytes_arr_automon.pkl", 'rb'))
        periodic_message_size = pickle.load(open(aws_result_folder + "periodic_message_size.pkl", 'rb'))
        num_messages_arr_periodic = pickle.load(open(aws_result_folder + "num_messages_arr_periodic.pkl", 'rb'))
        max_error_arr_periodic = pickle.load(open(aws_result_folder + "max_error_arr_periodic.pkl", 'rb'))
        aws_num_bytes_arr_zmq = pickle.load(open(aws_result_folder + "aws_num_bytes_arr_zmq.pkl", 'rb'))
        centralization_transfer_volume = pickle.load(open(aws_result_folder + "centralization_transfer_volume.pkl", 'rb'))
        num_bytes_arr_periodic = pickle.load(open(aws_result_folder + "num_bytes_arr_periodic.pkl", 'rb'))
        centralization_num_messages = pickle.load(open(aws_result_folder + "centralization_num_messages.pkl", 'rb'))
        sim_error_bound_arr = pickle.load(open(aws_result_folder + "sim_error_bound_arr.pkl", 'rb'))
        sim_num_messages_arr_automon = pickle.load(open(aws_result_folder + "sim_num_messages_arr_automon.pkl", 'rb'))
        sim_max_error_arr_automon = pickle.load(open(aws_result_folder + "sim_max_error_arr_automon.pkl", 'rb'))

    if b_print:
        if "dnn" in function:
            percent_diff_distributed_to_simulation = (np.array(aws_num_messages_arr_automon) - np.array(sim_num_messages_arr_automon)[[1, 3, 4, 5, 7, 9]]) / np.array(sim_num_messages_arr_automon)[[1, 3, 4, 5, 7, 9]] * 100
        elif "kld" in function:
            percent_diff_distributed_to_simulation = (np.array(aws_num_messages_arr_automon) - np.array(sim_num_messages_arr_automon)[[2, 3, 4, 5, 6, 7, 8, 10]]) / np.array(sim_num_messages_arr_automon)[[2, 3, 4, 5, 6, 7, 8, 10]] * 100
        else:
            percent_diff_distributed_to_simulation = (np.array(aws_num_messages_arr_automon) - sim_num_messages_arr_automon) / sim_num_messages_arr_automon * 100
        print(function, "median of differences between simulation total payload and distributed experiment total payload:", np.median(percent_diff_distributed_to_simulation), "%")

    # Remove points to the left of Centralization vertical line
    limit = centralization_num_messages

    args = np.argwhere(np.array(num_messages_arr_periodic) <= limit).squeeze()
    num_messages_arr_periodic = np.take(num_messages_arr_periodic, args)
    max_error_arr_periodic = np.take(max_error_arr_periodic, args)
    num_bytes_arr_periodic = np.take(num_bytes_arr_periodic, args)

    args = np.argwhere(np.array(aws_num_messages_arr_automon) <= limit).squeeze()
    if len(args.shape) == 0:
        args = np.expand_dims(args, axis=-1)
    aws_error_bound_arr = np.take(aws_error_bound_arr, args)
    aws_num_messages_arr_automon = np.take(aws_num_messages_arr_automon, args)
    aws_max_error_arr_automon = np.take(aws_max_error_arr_automon, args)
    aws_num_bytes_arr_automon = np.take(aws_num_bytes_arr_automon, args)
    aws_num_bytes_arr_zmq = np.take(aws_num_bytes_arr_zmq, args)

    args = np.argwhere(np.array(sim_num_messages_arr_automon) <= limit).squeeze()
    sim_error_bound_arr = np.take(sim_error_bound_arr, args)
    sim_num_messages_arr_automon = np.take(sim_num_messages_arr_automon, args)
    sim_max_error_arr_automon = np.take(sim_max_error_arr_automon, args)

    return aws_error_bound_arr, aws_num_messages_arr_automon, aws_max_error_arr_automon, real_function_value, num_nodes, aws_num_bytes_arr_automon, aws_num_bytes_arr_zmq, \
           periodic_message_size, num_messages_arr_periodic, max_error_arr_periodic, \
           centralization_transfer_volume, num_bytes_arr_periodic, centralization_num_messages, sim_error_bound_arr, sim_num_messages_arr_automon, sim_max_error_arr_automon


def plot_max_error_vs_transfer_volume(parent_test_folder_sim, parent_test_folder_aws, parent_test_folder_centralization, function, result_dir="./"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    aws_error_bound_arr, aws_num_messages_arr_automon, aws_max_error_arr_automon, real_function_value, num_nodes, aws_num_bytes_arr_automon, aws_num_bytes_arr_zmq, \
    periodic_message_size, num_messages_arr_periodic, max_error_arr_periodic, \
    centralization_transfer_volume, num_bytes_arr_periodic, centralization_num_messages, sim_error_bound_arr, sim_num_messages_arr_automon, sim_max_error_arr_automon = \
        read_data(parent_test_folder_sim, parent_test_folder_aws, parent_test_folder_centralization, function, b_print=True)
    centralization_transfer_volume_wo_overhead = periodic_message_size * centralization_num_messages

    # Figure with error as a function of number of messages - simulation and AWS

    fig, axs = plt.subplots(1, 2, figsize=get_figsize(hf=0.3), sharey=True)
    ax = axs[0]
    ax.set_ylabel('max error')
    ax.set_xlabel('#messages')
    ax.plot(sim_num_messages_arr_automon, sim_max_error_arr_automon, label="AutoMon sim.", marker="x", markersize=3, color="tab:blue", linewidth=0.8)
    ax.plot(aws_num_messages_arr_automon, aws_max_error_arr_automon, label="AutoMon dist.", marker="+", markersize=3, color="tab:brown", linestyle=(0, (1, 0.5)), linewidth=0.8)
    ax.plot(num_messages_arr_periodic, max_error_arr_periodic, label="Periodic", linestyle="--", marker=".", markersize=5, color="tab:green", linewidth=0.8)
    ax.axvline(x=centralization_num_messages, color="black", linestyle=":", label="Centralization", linewidth=0.8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    # Figure with error as a function of number of bytes - simulation and AWS

    MB = 2**20
    ax = axs[1]
    ax.set_xlabel('total size (MB)')
    ax.plot(aws_num_bytes_arr_automon / MB, aws_max_error_arr_automon, label="AutoMon dist.", marker="+", markersize=3, color="tab:brown", linestyle=(0, (1, 0.5)), linewidth=0.8)
    ax.plot(num_bytes_arr_periodic / MB, max_error_arr_periodic, label="Periodic", linestyle="--", marker=".", markersize=5, color="tab:green", linewidth=0.8)
    ax.axvline(x=centralization_num_messages * periodic_message_size / MB, color="black", linestyle=":", label="Centralization", linewidth=0.8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    handles, labels = axs[0].get_legend_handles_labels()
    plt.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(-0.2, 1.33), columnspacing=1.5, handletextpad=0.6, frameon=False, framealpha=0, handlelength=1.5)
    plt.subplots_adjust(top=0.88, bottom=0.32, left=0.11, right=0.97, wspace=0.1)
    fig.savefig(result_dir + "/max_error_vs_communication_sim_and_aws_" + function + ".pdf")
    plt.close(fig)

    # Figure stacked barchart of bytes per error bound - at the bottom AutoMon bytes and on top network bytes (ZMQ + TCP)

    fig, ax = plt.subplots(figsize=get_figsize(hf=0.4))
    ax.set_ylabel('total size (MB)')
    #ax.set_xlabel(r'error bound $\epsilon$')
    ax.set_xlabel('error bound \u03B5')

    bar_width = 0.8
    index = np.arange(len(aws_error_bound_arr))
    ax.bar(index, np.array(aws_num_bytes_arr_automon)/MB, bar_width, label="AutoMon", color='tab:blue', edgecolor='black', linewidth=0.2)
    ax.bar(index, np.array(aws_num_bytes_arr_zmq)/MB - np.array(aws_num_bytes_arr_automon)/MB, bar_width, label="network",
           color='tab:orange', edgecolor='black', linewidth=0.2, bottom=aws_num_bytes_arr_automon/MB)
    plt.xticks(index, [str(error_bound).lstrip("0") for error_bound in aws_error_bound_arr])
    ax.set_xlim([-0.6*bar_width, len(aws_error_bound_arr)-0.5*bar_width])

    # Centralization transfer volume (MB)
    ax.plot([-0.6 * bar_width, len(aws_error_bound_arr) - 0.5 * bar_width], [centralization_transfer_volume / MB, centralization_transfer_volume / MB], "--", linewidth=0.7, color="black", label="Central. traffic")
    # Data size (equals Centralization transfer volume without network overhead (MB))
    ax.plot([-0.6 * bar_width, len(aws_error_bound_arr) - 0.5 * bar_width], [centralization_transfer_volume_wo_overhead / MB, centralization_transfer_volume_wo_overhead / MB], ":", linewidth=0.7, color="black", label="Central. payload")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    ax.legend(frameon=False, ncol=2)
    plt.subplots_adjust(top=0.9, bottom=0.3, left=0.16, right=0.99)
    fig.savefig(result_dir + "/communication_automon_vs_network_" + function + ".pdf")
    plt.close(fig)

    rcParams.update(rcParamsDefault)


def plot_max_error_vs_transfer_volume_single(ax, func_name, aws_num_bytes_arr_automon, aws_max_error_arr_automon,
                                             num_bytes_arr_periodic, max_error_arr_periodic, centralization_num_messages, periodic_message_size):
    MB = 2**20
    ax.set_xlabel('total size (MB)', labelpad=3)

    ax.plot(aws_num_bytes_arr_automon / MB, aws_max_error_arr_automon, label="AutoMon", marker="+", markersize=3, color="tab:blue", linestyle=(0, (1, 0.5)), linewidth=0.8)
    ax.plot(num_bytes_arr_periodic / MB, max_error_arr_periodic, label="Periodic", linestyle="--", marker=".", markersize=3, color="tab:green", linewidth=0.8)
    ax.axvline(x=centralization_num_messages * periodic_message_size / MB, color="black", linestyle=":", label="Centralization", linewidth=0.8)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['left'].set_linewidth(0.3)
    ax.tick_params(width=0.3)
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_remove_leading_zeros))

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.tick_params(axis='y', which='major', pad=1)
    ax.tick_params(axis='x', which='major', pad=1.5)
    ax.set_title(func_name, pad=3)


def plot_max_error_vs_transfer_volume_combined(parent_test_folder_sim_kld, parent_test_folder_sim_inner_prod, parent_test_folder_sim_quadratic, parent_test_folder_sim_dnn,
                                               parent_test_folder_aws_kld, parent_test_folder_aws_inner_prod, parent_test_folder_aws_quadratic, parent_test_folder_aws_dnn,
                                               parent_test_folder_centralization_kld, parent_test_folder_centralization_inner_prod, parent_test_folder_centralization_quadratic, parent_test_folder_centralization_dnn,
                                               result_dir="./"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

    kld_aws_error_bound_arr, kld_aws_num_messages_arr_automon, kld_aws_max_error_arr_automon, kld_real_function_value, kld_num_nodes, kld_aws_num_bytes_arr_automon, kld_aws_num_bytes_arr_zmq, \
    kld_periodic_message_size, kld_num_messages_arr_periodic, kld_max_error_arr_periodic, \
    kld_centralization_transfer_volume, kld_num_bytes_arr_periodic, kld_centralization_num_messages, kld_sim_error_bound_arr, kld_sim_num_messages_arr_automon, kld_sim_max_error_arr_automon = \
        read_data(parent_test_folder_sim_kld, parent_test_folder_aws_kld, parent_test_folder_centralization_kld, "kld")
    inner_prod_aws_error_bound_arr, inner_prod_aws_num_messages_arr_automon, inner_prod_aws_max_error_arr_automon, inner_prod_real_function_value, inner_prod_num_nodes, inner_prod_aws_num_bytes_arr_automon, inner_prod_aws_num_bytes_arr_zmq, \
    inner_prod_periodic_message_size, inner_prod_num_messages_arr_periodic, inner_prod_max_error_arr_periodic, \
    inner_prod_centralization_transfer_volume, inner_prod_num_bytes_arr_periodic, inner_prod_centralization_num_messages, inner_prod_sim_error_bound_arr, inner_prod_sim_num_messages_arr_automon, inner_prod_sim_max_error_arr_automon = \
        read_data(parent_test_folder_sim_inner_prod, parent_test_folder_aws_inner_prod, parent_test_folder_centralization_inner_prod, "inner_product")
    quadratic_aws_error_bound_arr, quadratic_aws_num_messages_arr_automon, quadratic_aws_max_error_arr_automon, quadratic_real_function_value, quadratic_num_nodes, quadratic_aws_num_bytes_arr_automon, quadratic_aws_num_bytes_arr_zmq, \
    quadratic_periodic_message_size, quadratic_num_messages_arr_periodic, quadratic_max_error_arr_periodic, \
    quadratic_centralization_transfer_volume, quadratic_num_bytes_arr_periodic, quadratic_centralization_num_messages, quadratic_sim_error_bound_arr, quadratic_sim_num_messages_arr_automon, quadratic_sim_max_error_arr_automon = \
        read_data(parent_test_folder_sim_quadratic, parent_test_folder_aws_quadratic, parent_test_folder_centralization_quadratic, "quadratic")
    dnn_aws_error_bound_arr, dnn_aws_num_messages_arr_automon, dnn_aws_max_error_arr_automon, dnn_real_function_value, dnn_num_nodes, dnn_aws_num_bytes_arr_automon, dnn_aws_num_bytes_arr_zmq, \
    dnn_periodic_message_size, dnn_num_messages_arr_periodic, dnn_max_error_arr_periodic, \
    dnn_centralization_transfer_volume, dnn_num_bytes_arr_periodic, dnn_centralization_num_messages, dnn_sim_error_bound_arr, dnn_sim_num_messages_arr_automon, dnn_sim_max_error_arr_automon = \
        read_data(parent_test_folder_sim_dnn, parent_test_folder_aws_dnn, parent_test_folder_centralization_dnn, "dnn")

    # Figure with error as a function of transfer volume for 4 functions - inner product, quadratic, kld, dnn

    fig, axs = plt.subplots(1, 4, figsize=get_figsize(hf=0.3))
    axs[0].set_ylabel('max error', labelpad=2)

    plot_max_error_vs_transfer_volume_single(axs[0], "Inner Product", inner_prod_aws_num_bytes_arr_automon, inner_prod_aws_max_error_arr_automon,
                                             inner_prod_num_bytes_arr_periodic, inner_prod_max_error_arr_periodic,
                                             inner_prod_centralization_num_messages, inner_prod_periodic_message_size)
    plot_max_error_vs_transfer_volume_single(axs[1], "Quadratic", quadratic_aws_num_bytes_arr_automon,
                                             quadratic_aws_max_error_arr_automon,
                                             quadratic_num_bytes_arr_periodic, quadratic_max_error_arr_periodic,
                                             quadratic_centralization_num_messages, quadratic_periodic_message_size)
    plot_max_error_vs_transfer_volume_single(axs[2], "KLD", kld_aws_num_bytes_arr_automon,
                                             kld_aws_max_error_arr_automon,
                                             kld_num_bytes_arr_periodic, kld_max_error_arr_periodic,
                                             kld_centralization_num_messages, kld_periodic_message_size)
    plot_max_error_vs_transfer_volume_single(axs[3], "DNN", dnn_aws_num_bytes_arr_automon,
                                             dnn_aws_max_error_arr_automon,
                                             dnn_num_bytes_arr_periodic, dnn_max_error_arr_periodic,
                                             dnn_centralization_num_messages, dnn_periodic_message_size)

    axs[3].set_yticks([0, 0.02, 0.04])
    axs[0].set_xticks([0, 1, 2, 3])
    axs[1].set_xticks([0, 1, 2, 3])
    axs[2].set_xticks([0, 25, 50])
    axs[3].set_xticks([0, 50, 100])

    handles, labels = axs[0].get_legend_handles_labels()
    plt.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(-1.5, -0.59), columnspacing=1.5,
               handletextpad=0.6, frameon=False, framealpha=0, handlelength=1.5)
    plt.subplots_adjust(top=0.88, bottom=0.41, left=0.07, right=0.98, wspace=0.4)
    fig.savefig(result_dir + "/max_error_vs_transfer_volume.pdf")
    plt.close(fig)

    rcParams.update(rcParamsDefault)


def reformat_remove_leading_zeros(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x)
    if 0 < np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str


def plot_communication_automon_vs_network(ax, func_name, aws_error_bound_arr, aws_num_bytes_arr_automon, aws_num_bytes_arr_zmq, centralization_transfer_volume, chosen_error_bounds, centralization_transfer_volume_wo_overhead):
    MB = 2**20

    if "KLD" in func_name or "DNN" in func_name:
        #ax.set_xlabel(r'error bound $\epsilon$', labelpad=1)
        ax.set_xlabel('error bound \u03B5', labelpad=1)
    else:
        #ax.set_xlabel(r'error bound $\epsilon$', labelpad=3)
        ax.set_xlabel('error bound \u03B5', labelpad=3)

    transfer_volume_reduction = np.array(centralization_transfer_volume) - np.array(aws_num_bytes_arr_zmq)
    print(func_name, "data transfer volume reduction (compared to Centralization transfer volume) "
          "max:", np.max(transfer_volume_reduction) / centralization_transfer_volume * 100, "%, "
          "avg:", np.mean(transfer_volume_reduction) / centralization_transfer_volume * 100, "%")

    indices = np.where(np.in1d(aws_error_bound_arr, chosen_error_bounds))[0]
    aws_error_bound_arr = np.take(np.array(aws_error_bound_arr), indices)
    aws_num_bytes_arr_automon = np.take(np.array(aws_num_bytes_arr_automon), indices) / MB
    aws_num_bytes_arr_zmq = np.take(np.array(aws_num_bytes_arr_zmq), indices) / MB

    bar_width = 0.8
    index = np.arange(len(aws_error_bound_arr))
    ax.bar(index, aws_num_bytes_arr_automon, bar_width, label="AutoMon payload", color='tab:blue', edgecolor='none', linewidth=0.2)
    ax.bar(index, np.array(aws_num_bytes_arr_zmq) - np.array(aws_num_bytes_arr_automon), bar_width, label="AutoMon traffic", color='tab:orange', edgecolor='none', linewidth=0.2, bottom=aws_num_bytes_arr_automon)
    ax.set_xticks(index)
    ax.set_xticklabels([str(error_bound).lstrip("0") for error_bound in aws_error_bound_arr], rotation=50)
    ax.tick_params(axis='x', which='major', pad=1)
    ax.set_xlim([-0.6 * bar_width, len(aws_error_bound_arr) - 0.5 * bar_width])

    # Centralization transfer volume (MB)
    ax.plot([-0.6 * bar_width, len(aws_error_bound_arr) - 0.5 * bar_width], [centralization_transfer_volume/MB, centralization_transfer_volume/MB], "--", linewidth=0.7, color="black", label="Central. traffic")
    # Data size (equals Centralization transfer volume without network overhead (MB))
    ax.plot([-0.6 * bar_width, len(aws_error_bound_arr) - 0.5 * bar_width], [centralization_transfer_volume_wo_overhead/MB, centralization_transfer_volume_wo_overhead/MB], ":", linewidth=0.7, color="black", label="Central. payload")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['left'].set_linewidth(0.3)
    ax.tick_params(width=0.3)
    ax.tick_params(axis='y', which='major', pad=1)
    ax.set_title(func_name, pad=3)


def plot_communication_automon_vs_network_combined(parent_test_folder_sim_kld, parent_test_folder_sim_inner_prod, parent_test_folder_sim_quadratic, parent_test_folder_sim_dnn,
                                                   parent_test_folder_aws_kld, parent_test_folder_aws_inner_prod, parent_test_folder_aws_quadratic, parent_test_folder_aws_dnn,
                                                   parent_test_folder_centralization_kld, parent_test_folder_centralization_inner_prod, parent_test_folder_centralization_quadratic, parent_test_folder_centralization_dnn,
                                                   result_dir="./"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

    kld_aws_error_bound_arr, _, _, _, _, kld_aws_num_bytes_arr_automon, kld_aws_num_bytes_arr_zmq, periodic_message_size, _, _, kld_centralization_transfer_volume, _, centralization_num_messages, _, _, _ = read_data(
        parent_test_folder_sim_kld, parent_test_folder_aws_kld, parent_test_folder_centralization_kld, "kld")
    kld_centralization_transfer_volume_wo_overhead = periodic_message_size * centralization_num_messages
    inner_prod_aws_error_bound_arr, _, _, _, _, inner_prod_aws_num_bytes_arr_automon, inner_prod_aws_num_bytes_arr_zmq, periodic_message_size, _, _, inner_prod_centralization_transfer_volume, _, centralization_num_messages, _, _, _ = read_data(
        parent_test_folder_sim_inner_prod, parent_test_folder_aws_inner_prod, parent_test_folder_centralization_inner_prod, "inner_product")
    inner_prod_centralization_transfer_volume_wo_overhead = periodic_message_size * centralization_num_messages
    quadratic_aws_error_bound_arr, _, _, _, _, quadratic_aws_num_bytes_arr_automon, quadratic_aws_num_bytes_arr_zmq, periodic_message_size, _, _, quadratic_centralization_transfer_volume, _, centralization_num_messages, _, _, _ = read_data(
        parent_test_folder_sim_quadratic, parent_test_folder_aws_quadratic, parent_test_folder_centralization_quadratic, "quadratic")
    quadratic_centralization_transfer_volume_wo_overhead = periodic_message_size * centralization_num_messages
    dnn_aws_error_bound_arr, _, _, _, _, dnn_aws_num_bytes_arr_automon, dnn_aws_num_bytes_arr_zmq, periodic_message_size, _, _, dnn_centralization_transfer_volume, _, centralization_num_messages, _, _, _ = read_data(
        parent_test_folder_sim_dnn, parent_test_folder_aws_dnn, parent_test_folder_centralization_dnn, "dnn")
    dnn_centralization_transfer_volume_wo_overhead = periodic_message_size * centralization_num_messages

    # Figure stacked barchart of bytes per error bound - at the bottom AutoMon bytes and on top network bytes (ZMQ + TCP)

    fig, axs = plt.subplots(1, 4, figsize=get_figsize(hf=0.32))
    axs[0].set_ylabel('total size (MB)')

    plot_communication_automon_vs_network(axs[0], "Inner Product", inner_prod_aws_error_bound_arr, inner_prod_aws_num_bytes_arr_automon,
                                          inner_prod_aws_num_bytes_arr_zmq, inner_prod_centralization_transfer_volume, [0.05, 0.1, 0.2, 0.8],
                                          inner_prod_centralization_transfer_volume_wo_overhead)
    plot_communication_automon_vs_network(axs[1], "Quadratic", quadratic_aws_error_bound_arr, quadratic_aws_num_bytes_arr_automon,
                                          quadratic_aws_num_bytes_arr_zmq, quadratic_centralization_transfer_volume, [0.03, 0.04, 0.08, 1.0],
                                          quadratic_centralization_transfer_volume_wo_overhead)
    plot_communication_automon_vs_network(axs[2], "KLD", kld_aws_error_bound_arr, kld_aws_num_bytes_arr_automon,
                                          kld_aws_num_bytes_arr_zmq, kld_centralization_transfer_volume, [0.005, 0.01, 0.02, 0.08],
                                          kld_centralization_transfer_volume_wo_overhead)
    plot_communication_automon_vs_network(axs[3], "DNN", dnn_aws_error_bound_arr, dnn_aws_num_bytes_arr_automon,
                                          dnn_aws_num_bytes_arr_zmq, dnn_centralization_transfer_volume, [0.002, 0.005, 0.007, 0.016],
                                          dnn_centralization_transfer_volume_wo_overhead)

    axs[0].set_yticks([0, 2, 4])
    axs[1].set_yticks([0, 2, 4])
    axs[2].set_yticks([0, 50, 100])
    axs[3].set_yticks([0, 50, 100, 150])

    handles, labels = axs[0].get_legend_handles_labels()
    handles = [handles[2], handles[3], handles[1], handles[0]]
    labels = [labels[2], labels[3], labels[1], labels[0]]
    plt.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(-1.95, -0.71), columnspacing=1.4, handletextpad=0.4, frameon=False, framealpha=0, handlelength=1.5)
    plt.subplots_adjust(top=0.9, bottom=0.45, left=0.08, right=0.99, wspace=0.5)
    fig.savefig(result_dir + "/communication_automon_vs_network.pdf")
    plt.close(fig)

    rcParams.update(rcParamsDefault)


def check_rtt_between_aws_regions(parent_test_folder_aws_kld, parent_test_folder_aws_inner_prod, parent_test_folder_aws_quadratic, parent_test_folder_aws_dnn, result_dir="./"):
    test_folders = get_all_test_folders(parent_test_folder_aws_kld)
    test_folders += get_all_test_folders(parent_test_folder_aws_inner_prod)
    test_folders += get_all_test_folders(parent_test_folder_aws_quadratic)
    test_folders += get_all_test_folders(parent_test_folder_aws_dnn)

    rtt_arr_all = []

    for test_folder in test_folders:
        if "coordinator" in test_folder:
            log_lines = []
            rtt_arr = []
            log_file = [test_folder + "/" + f for f in os.listdir(test_folder) if ".log" in f][0]
            with open(log_file, 'r') as f:
                log = f.read().split("\n")
                for log_line in log:
                    if "Coordinator asks node 0 for statistics" in log_line or "Node 0 notify violation" in log_line or "Node 0 returns to coordinator with statistics" in log_line:
                        log_lines.append(log_line)
            # Go over the lines. For every "Coordinator asks node 0 for statistics" line search its following "Node 0 notify violation" or "Node 0 returns to coordinator with statistics" and get their time diff
            for i in range(len(log_lines)):
                if "Coordinator asks node 0 for statistics" in log_lines[i]:
                    if "Node 0 returns to coordinator with statistics" not in log_lines[i + 1]:
                        continue
                    start = log_lines[i].split(" INFO")[0]
                    start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S,%f")
                    end = log_lines[i + 1].split(" INFO")[0]
                    end = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S,%f")
                    rtt = (end - start).total_seconds() * 1000
                    rtt_arr.append(rtt)
                    rtt_arr_all.append(rtt)
            if len(rtt_arr) == 0:
                continue

    avg_rtt = np.mean(rtt_arr_all)
    max_rtt = np.max(rtt_arr_all)
    median_rtt = np.median(rtt_arr_all)
    std_rtt = np.std(rtt_arr_all)
    print("RTT (round trip time) between the AWS regions average (ms):", avg_rtt, ", max RTT (ms):", max_rtt, ", median RTT (ms):", median_rtt, ", std (ms):", std_rtt)


if __name__ == "__main__":
    # Figure 5 and Figure 6

    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
        relative_path = '../'

        inner_product_test_folder = result_dir + "/" + sys.argv[2]
        kld_test_folder = result_dir + "/" + sys.argv[3]
        mlp_test_folder = result_dir + "/" + sys.argv[4]

        parent_test_folder_inner_prod_sim = result_dir + "/" + sys.argv[2]
        parent_test_folder_quadratic_sim = result_dir + "/" + sys.argv[3]
        parent_test_folder_kld_sim = result_dir + "/" + sys.argv[4]
        parent_test_folder_dnn_sim = result_dir + "/" + sys.argv[5]

        parent_test_folder_inner_prod_aws = result_dir + "/" + sys.argv[6]
        parent_test_folder_quadratic_aws = result_dir + "/" + sys.argv[7]
        parent_test_folder_kld_aws = result_dir + "/" + sys.argv[8]
        parent_test_folder_dnn_aws = result_dir + "/" + sys.argv[9]

        parent_test_folder_inner_prod_centralization = result_dir + "/" + sys.argv[10]
        parent_test_folder_quadratic_centralization = result_dir + "/" + sys.argv[11]
        parent_test_folder_kld_centralization = result_dir + "/" + sys.argv[12]
        parent_test_folder_dnn_centralization = result_dir + "/" + sys.argv[13]
    else:
        result_dir = "./"
        relative_path = '../../'

        parent_test_folder_inner_prod_sim = "../test_results/results_test_max_error_vs_communication_inner_product_2021-09-29_05-42-23"
        parent_test_folder_quadratic_sim = "../test_results/results_test_max_error_vs_communication_quadratic_2021-09-29_05-38-15"
        parent_test_folder_kld_sim = "../test_results/results_test_max_error_vs_communication_kld_air_quality_2021-09-19_23-02-36"
        parent_test_folder_dnn_sim = "../test_results/results_test_max_error_vs_communication_dnn_intrusion_detection_2021-09-19_23-05-20"

        parent_test_folder_inner_prod_aws = "../../aws_experiments/test_results/max_error_vs_communication_inner_product_aws_2021-10-10"
        parent_test_folder_quadratic_aws = "../../aws_experiments/test_results/max_error_vs_communication_quadratic_aws_2021-10-10"
        parent_test_folder_kld_aws = "../../aws_experiments/test_results/max_error_vs_communication_kld_aws_2021-10-26"
        parent_test_folder_dnn_aws = "../../aws_experiments/test_results/max_error_vs_communication_dnn_aws_2021-10-26"

        parent_test_folder_inner_prod_centralization = "../test_results/results_dist_centralization_inner_product_2021-10-18_19-47-14"
        parent_test_folder_quadratic_centralization = "../test_results/results_dist_centralization_quadratic_2021-10-18_19-51-19"
        parent_test_folder_kld_centralization = "../test_results/results_dist_centralization_kld_2021-10-18_20-01-13"
        parent_test_folder_dnn_centralization = "../test_results/results_dist_centralization_dnn_2021-10-18_20-16-53"

    # Remote

    plot_max_error_vs_transfer_volume(parent_test_folder_kld_sim, parent_test_folder_kld_aws, parent_test_folder_kld_centralization, "kld", result_dir)
    plot_max_error_vs_transfer_volume(parent_test_folder_inner_prod_sim, parent_test_folder_inner_prod_aws, parent_test_folder_inner_prod_centralization, "inner_product", result_dir)
    plot_max_error_vs_transfer_volume(parent_test_folder_quadratic_sim, parent_test_folder_quadratic_aws, parent_test_folder_quadratic_centralization, "quadratic", result_dir)
    plot_max_error_vs_transfer_volume(parent_test_folder_dnn_sim, parent_test_folder_dnn_aws, parent_test_folder_dnn_centralization, "dnn", result_dir)

    # Figure 10 (bottom)
    plot_communication_automon_vs_network_combined(parent_test_folder_kld_sim, parent_test_folder_inner_prod_sim, parent_test_folder_quadratic_sim, parent_test_folder_dnn_sim,
                                                   parent_test_folder_kld_aws, parent_test_folder_inner_prod_aws, parent_test_folder_quadratic_aws, parent_test_folder_dnn_aws,
                                                   parent_test_folder_kld_centralization, parent_test_folder_inner_prod_centralization, parent_test_folder_quadratic_centralization, parent_test_folder_dnn_centralization,
                                                   result_dir)

    # Figure 10 (top)
    plot_max_error_vs_transfer_volume_combined(parent_test_folder_kld_sim, parent_test_folder_inner_prod_sim, parent_test_folder_quadratic_sim, parent_test_folder_dnn_sim,
                                               parent_test_folder_kld_aws, parent_test_folder_inner_prod_aws, parent_test_folder_quadratic_aws, parent_test_folder_dnn_aws,
                                               parent_test_folder_kld_centralization, parent_test_folder_inner_prod_centralization, parent_test_folder_quadratic_centralization, parent_test_folder_dnn_centralization,
                                               result_dir)

    check_rtt_between_aws_regions(parent_test_folder_kld_aws, parent_test_folder_inner_prod_aws, parent_test_folder_quadratic_aws, parent_test_folder_dnn_aws, result_dir)
