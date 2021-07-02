from coordinators.coordinator_common import SlackType, SyncType
from coordinators.coordinator_auto_mon import DomainType
import os
import datetime
from timeit import default_timer as timer
import argparse
import logging
from importlib import reload
import numpy as np
import sys

from node_stream import NodeStream


def _prepare_test_folder(test_name, test_folder=""):
    if test_folder == "":
        test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        test_folder = os.path.join(os.getcwd(), 'test_results/results_' + test_name + "_" + test_timestamp)
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)
    reload(logging)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    output_file_handler = logging.FileHandler(test_folder + "/" + test_name + '.log', mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    logging.getLogger('matplotlib.backends.backend_pdf').setLevel(logging.WARNING)
    logging.info('Started test - ' + test_name)
    return test_folder


def start_test(test_name, test_folder=""):
    test_folder = _prepare_test_folder(test_name, test_folder)
    return test_folder


def end_test():
    logging.info('Finished test')
    logging.shutdown()
    reload(logging)


def _test_data_loop(coordinator, nodes, verifier, data_generator, sliding_window_size, func_update_local_vector, b_check_violation_every_sample):
    local_vec_len = 1 if len(data_generator.data.shape) == 1 else data_generator.data.shape[1]
    num_nodes = len(nodes)
    node_stream = NodeStream(num_nodes, sliding_window_size, local_vec_len, func_update_local_vector, verifier.x0)
    num_iterations = data_generator.get_num_samples()

    # Fill all sliding windows
    num_samples_to_fill_windows = 0
    while not node_stream.all_windows_full():
        if b_check_violation_every_sample:
            data_point_and_node_idx = data_generator.get_next_data_point()
            node_idx = int(data_point_and_node_idx[0])
            data_point = data_point_and_node_idx[1:]
        else:
            node_idx = num_samples_to_fill_windows % num_nodes
            data_point = data_generator.get_next_data_point()
        node_stream.set_new_data_point(data_point, node_idx)
        num_samples_to_fill_windows += 1

    assert (num_samples_to_fill_windows == num_nodes * sliding_window_size)

    # First iteration after all sliding windows are full: provide the verifier the global vector, provide all the
    # nodes their local vectors, and call the coordinator to resolve violations.
    global_vector = node_stream.get_global_vector()
    verifier.set_new_data_point(global_vector)
    for node_idx, node in enumerate(nodes):
        local_vector = node_stream.get_local_vector(node_idx)
        b_node_inside_safe_zone = node.set_new_data_point(local_vector)
    coordinator.sync_if_needed()

    # For the rest of the iterations: read data from stream, update verifier global vector and nodes local
    # vectors, and call the coordinator to resolve violations.
    iteration_counter = 0
    num_iterations -= num_samples_to_fill_windows
    for i in range(num_iterations):
        if b_check_violation_every_sample:
            data_point_and_node_idx = data_generator.get_next_data_point()
            node_idx = int(data_point_and_node_idx[0])
            data_point = data_point_and_node_idx[1:]
        else:
            node_idx = i % num_nodes
            data_point = data_generator.get_next_data_point()
        node_stream.set_new_data_point(data_point, node_idx)

        local_vector = node_stream.get_local_vector(node_idx)
        global_vector = node_stream.get_global_vector()

        # Update the verifier first with the global vector, so the coordinator can check, after the update
        # of the node, if the violation is "false local", "false global" or "true".
        verifier.set_new_data_point(global_vector)
        b_node_inside_safe_zone = nodes[node_idx].set_new_data_point(local_vector)

        if b_check_violation_every_sample:
            coordinator.sync_if_needed()
        else:
            if i % num_nodes == num_nodes - 1:
                coordinator.sync_if_needed()
                iteration_counter += 1

    if not b_check_violation_every_sample:
        assert(iteration_counter == data_generator.num_iterations - (num_samples_to_fill_windows // num_nodes))


def run_test(data_generator, coordinator, nodes, verifier, test_folder, sliding_window_size, func_update_local_vector, b_check_violation_every_sample=False):
    logging.info("num_nodes " + str(len(nodes)) + ", num_iterations " + str(data_generator.num_iterations))
    
    coordinator.set_nodes(nodes)
    for node in nodes:
        node.set_coordinator(coordinator)
    
    # Send data to nodes
    start = timer()
    _test_data_loop(coordinator, nodes, verifier, data_generator, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
    end = timer()
    logging.info("The test took: " + str(end - start) + " seconds")
    full_sync_history, msg_counters = coordinator.dump_stats(test_folder)

    for node in nodes:
        node.dump_stats()
    
    return full_sync_history, msg_counters


def get_config(num_nodes=10, num_iterations=100, sliding_window_size=100, k=10, error_bound=0.05,
               slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
               domain=None, domain_type=DomainType.Absolute.value, neighborhood_size=1.0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, dest="num_nodes", help="num nodes", default=num_nodes)
    parser.add_argument("--num_iterations", type=int, dest="num_iterations", help="num iterations", default=num_iterations)
    parser.add_argument("--sliding_window_size", type=int, dest="sliding_window_size", help="sliding window size", default=sliding_window_size)
    parser.add_argument("--k", type=int, dest="k", help="dimention", default=k)
    parser.add_argument("--error_bound", type=float, dest="error_bound", help="additive error bound", default=error_bound)
    parser.add_argument("--slack_type", type=int, dest="slack_type", help="slak type - NoSlack (0) / Drift (1)", default=slack_type)
    parser.add_argument("--sync_type", type=int, dest="sync_type", help="sync type - Eager (0) / LazyRandom (1) / LazyLRU (2)", default=sync_type)
    parser.add_argument("--domain", type=tuple, dest="domain", help="1d domain. None for (-inf, inf) or (lower, upper), which are both not None. This (lower, upper) is applied to each coordinate of x.", default=domain)
    parser.add_argument("--domain_type", type=int, dest="domain_type", help="domain type - Absolute (0) / Relative (1)", default=domain_type)
    parser.add_argument("--neighborhood_size", type=int, dest="neighborhood_size", help="the neighborhood size if domain_type is Relative", default=neighborhood_size)
    conf = vars(parser.parse_args())
    return conf


def read_config_file(test_folder):
    with open(test_folder + "/config.txt") as conf_file:
        conf = conf_file.read()
    conf = eval(conf)
    return conf


def write_config_to_file(test_folder, conf):
    with open(test_folder + "/config.txt", 'w') as txt_file:
        txt_file.write(str(conf))
