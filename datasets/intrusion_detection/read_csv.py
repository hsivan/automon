import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import importlib.resources as pkg_resources


def get_data(file):
    df = pd.read_csv(file, header=None)

    # Change categorical columns to numerical values. Use the following mapping to keep the same mapping as in https://github.com/rahulvigneswaran/Intrusion-Detection-Systems

    # df[1]: transport layer protocol
    mapping = {'tcp': 1, 'udp': 2, 'icmp': 3}
    df = df.replace({1: mapping})
    # df[2]: application
    mapping = {'http': 1, 'smtp': 19, 'finger': 51, 'domain_u': 57, 'auth': 65, 'telnet': 13, 'ftp': 50, 'eco_i': 55,
               'ntp_u': 30, 'ecr_i': 54, 'other': 29, 'private': 24, 'pop_3': 26, 'ftp_data': 49, 'rje': 21, 'time': 10,
               'mtp': 38, 'link': 40, 'remote_job': 22, 'gopher': 48, 'ssh': 17, 'name': 37, 'whois': 4, 'domain': 58,
               'login': 39, 'imap4': 46, 'daytime': 70, 'ctf': 61, 'nntp': 31, 'shell': 20, 'IRC': 45, 'nnsp': 32,
               'http_443': 47, 'exec': 52, 'printer': 25, 'efs': 53, 'courier': 63, 'uucp': 7, 'klogin': 43, 'kshell': 42,
               'echo': 56, 'discard': 59, 'systat': 14, 'supdup': 15, 'iso_tsap': 44, 'hostnames': 69, 'csnet_ns': 62,
               'pop_2': 27, 'sunrpc': 16, 'uucp_path': 6, 'netbios_ns': 35, 'netbios_ssn': 34, 'netbios_dgm': 36, 'sql_net': 18,
               'vmnet': 5, 'bgp': 64, 'Z39_50': 2, 'ldap': 41, 'netstat': 33, 'urh_i': 9, 'X11': 3, 'urp_i': 8,
               'pm_dump': 28, 'tftp_u': 12, 'tim_i': 11, 'red_i': 23, 'icmp': 68}
    df = df.replace({2: mapping})
    # df[3]: connection flag
    mapping = {'SF': 1, 'SH': 2, 'S3': 3, 'OTH': 4, 'S2': 4, 'S1': 5, 'S0': 6, 'RSTR': 7, 'RSTOS0': 8, 'RSTO': 9, 'REJ': 10}
    df = df.replace({3: mapping})

    # Set label 'normal' to 0 and other labels to 1
    values = pd.unique(df[41])
    for value in values:
        if value == 'normal.':
            df[41].replace(value, 0, inplace=True)
        else:
            df[41].replace(value, 1, inplace=True)
    # Move last column to be the first
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    col_names = dict([(i, i+1) for i in range(41)])
    col_names[41] = 0
    df.rename(columns=col_names, inplace=True)
    return df


# Data taken from http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html kddcup.data_10_percent.gz
def get_training_data():
    data_stream = pkg_resources.open_text('datasets.intrusion_detection', 'kddcup.data_10_percent_corrected')
    return get_data(data_stream)


# Data taken from http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html corrected.gz
def get_testing_data():
    data_stream = pkg_resources.open_text('datasets.intrusion_detection', 'corrected')
    return get_data(data_stream)


def set_nodes_data(df_, app_idx, node_indices, nodes):
    apps, counts = np.unique(df_[3], return_counts=True)
    app_type = apps[app_idx]
    num_nodes = len(nodes)

    '''fig = plt.figure()
    tt = df_[df_[3].isin(app_type)]
    plt.scatter(tt.index, tt.index)
    plt.show()'''

    app_type_data = df_[df_[3].isin(app_type)]
    for i, node in enumerate(nodes):
        node_data = app_type_data[i::num_nodes]  # Round-robin between the nodes that get this application rows
        node_indices[node_data.index] = nodes[i]
    return node_indices


def prepare_intrusion_detection_data(num_rows_to_drop_from_beginning=0):
    sliding_window_size = 20
    num_nodes = 9

    df = get_testing_data()
    apps, counts = np.unique(df[3], return_counts=True)
    # Divide 164352 rows with application index 51 to 5 nodes (32870 rows per node).
    # Divide 78510 rows with application index 21 to 2 nodes (39255 rows per node).
    # Put 41237 rows with application index 0 on single node.
    # Put the rest 26930 rows on single node.
    # Overall 9 nodes with at least 30000 samples each.
    counts[51]  # 164352 (Private)
    counts[21]  # 78510 (ECR_i)
    counts[0]  # 41237 (http)

    node_indices = np.zeros(df.shape[0])

    train_data = get_training_data()
    train_x = train_data.iloc[:, 1:42].values
    scaler = Normalizer().fit(train_x)

    node_indices = set_nodes_data(df, [51], node_indices, [0, 1, 2, 3, 4])
    node_indices = set_nodes_data(df, [21], node_indices, [5, 6])
    node_indices = set_nodes_data(df, [0], node_indices, [7])
    rest_app_indices = [i for i in range(len(apps)) if i != 51 and i != 21 and i != 0]
    node_indices = set_nodes_data(df, rest_app_indices, node_indices, [8])[num_rows_to_drop_from_beginning:]

    nodes_data = df.values[num_rows_to_drop_from_beginning:][:, 1:42]
    nodes_data = scaler.transform(nodes_data)
    nodes_data = np.append(np.expand_dims(node_indices, -1), nodes_data, axis=1)
    max_index_of_full_window = 0
    for node_idx in range(num_nodes):
        index_of_full_window = np.argwhere(nodes_data[:, 0] == node_idx)[sliding_window_size][0]
        if index_of_full_window > max_index_of_full_window:
            max_index_of_full_window = index_of_full_window
    # For every node leave the last sliding_window_size samples before max_index_of_full_window in nodes_data
    num_rows_to_remove_per_node = []
    for node_idx in range(num_nodes):
        num_rows_to_remove = sum(np.argwhere(nodes_data[:, 0] == node_idx) < max_index_of_full_window)[0] - sliding_window_size
        num_rows_to_remove_per_node.append(num_rows_to_remove)
    all_rows_to_remove = []
    for node_idx in range(num_nodes):
        rows_to_remove = np.argwhere(nodes_data[:, 0] == node_idx).squeeze()[:num_rows_to_remove_per_node[node_idx]]
        all_rows_to_remove += rows_to_remove.tolist()
    nodes_data = np.delete(nodes_data, all_rows_to_remove, axis=0)
    # Take sliding_window_size*num_nodes first rows and make them interleaving: 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, ...
    initial_data = nodes_data[0:sliding_window_size*num_nodes].copy()
    for node_idx in range(num_nodes):
        initial_data[node_idx:sliding_window_size*num_nodes:num_nodes] = nodes_data[np.argwhere(nodes_data[0:sliding_window_size*num_nodes, 0] == node_idx).squeeze()]
    nodes_data[0:sliding_window_size*num_nodes] = initial_data

    return nodes_data


if __name__ == "__main__":
    prepare_intrusion_detection_data()
    get_training_data()
    get_testing_data()
