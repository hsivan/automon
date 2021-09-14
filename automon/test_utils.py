from automon.coordinator_common import SlackType, SyncType
from automon.automon.coordinator_automon import DomainType
import os
import datetime
from timeit import default_timer as timer
import logging
from importlib import reload
import sys
from automon.messages_common import prepare_message_data_update
from automon.node_stream import NodeStream


def _prepare_test_folder(test_name, test_folder=""):
    if test_folder == "":
        test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        test_folder = os.path.join(os.getcwd(), 'test_results/results_' + test_name + "_" + test_timestamp)
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)
    reload(logging)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Can change to DEBUG to see send and received message logs from messages_common
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s')
    output_file_handler = logging.FileHandler(test_folder + "/" + test_name + '.log', mode='w')
    output_file_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
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


def _from_violation_to_sync(coordinator, nodes, message_violations):
    messages_from_node_to_coordinator = message_violations
    while len(messages_from_node_to_coordinator) > 0:  # There are no more messages from the nodes after the coordinator sends sync
        messages_from_coordinator_to_nodes = coordinator.parse_message(messages_from_node_to_coordinator)
        messages_from_node_to_coordinator = b''
        for node_idx, message in messages_from_coordinator_to_nodes:
            message_from_node_to_coordinator = nodes[node_idx].parse_message(message)
            if message_from_node_to_coordinator is not None:
                messages_from_node_to_coordinator += message_from_node_to_coordinator


def _test_data_loop(coordinator, nodes, node_stream, data_generator, b_single_sample_per_round):
    # Fill all sliding windows
    num_samples_to_fill_windows = 0
    while not node_stream.all_windows_full():
        data_point, node_idx = data_generator.get_next_data_point()
        node_stream.set_new_data_point(data_point, node_idx)
        num_samples_to_fill_windows += 1

    assert (num_samples_to_fill_windows == node_stream.num_nodes * node_stream.sliding_window_size)

    # First data round after all sliding windows are full: provide the verifier the global vector, provide all the
    # nodes their local vectors, and call the coordinator to resolve violations.
    global_vector = node_stream.get_global_vector()
    coordinator.verifier.set_new_data_point(global_vector)  # Update the verifier
    message_violations = b''
    for node_idx, node in enumerate(nodes):
        local_vector = node_stream.get_local_vector(node_idx)
        message_data_update = prepare_message_data_update(node_idx, local_vector)
        message_violation = node.parse_message(message_data_update)
        if message_violation is not None and len(message_violation) > 0:
            message_violations += message_violation
    if len(message_violations) > 0:
        _from_violation_to_sync(coordinator, nodes, message_violations)
    coordinator.update_statistics()

    # For the rest of the data rounds: read data from stream, update verifier global vector and nodes local
    # vectors, and call the coordinator to resolve violations.
    message_violations = b''
    for i in range(data_generator.get_num_samples() - num_samples_to_fill_windows):
        data_point, node_idx = data_generator.get_next_data_point()
        node_stream.set_new_data_point(data_point, node_idx)

        local_vector = node_stream.get_local_vector(node_idx)
        global_vector = node_stream.get_global_vector()

        # Update the verifier first with the global vector, so the coordinator can check, after the update
        # of the node, if the violation is "false local", "false global" or "true".
        coordinator.verifier.set_new_data_point(global_vector)  # Update the verifier
        message_data_update = prepare_message_data_update(node_idx, local_vector)
        message_violation = nodes[node_idx].parse_message(message_data_update)
        if message_violation is not None and len(message_violation) > 0:
            message_violations += message_violation

        if b_single_sample_per_round:
            if len(message_violations) > 0:
                _from_violation_to_sync(coordinator, nodes, message_violations)
            coordinator.update_statistics()
            message_violations = b''
        else:
            if i % node_stream.num_nodes == node_stream.num_nodes - 1:
                if len(message_violations) > 0:
                    _from_violation_to_sync(coordinator, nodes, message_violations)
                coordinator.update_statistics()
                message_violations = b''


def run_test(data_generator, coordinator, nodes, test_folder, sliding_window_size, b_single_sample_per_round=False):
    logging.info("num_nodes " + str(len(nodes)) + ", num_iterations " + str(data_generator.get_num_iterations()) + ", data_generator state " + str(data_generator.state))

    node_stream = NodeStream(len(nodes), sliding_window_size, data_generator.get_data_point_len(), data_generator.get_local_vec_update_func(), initial_x0=data_generator.get_initial_x0())
    
    # Send data to nodes
    start = timer()
    _test_data_loop(coordinator, nodes, node_stream, data_generator, b_single_sample_per_round)
    end = timer()
    logging.info("The test took: " + str(end - start) + " seconds")
    full_sync_history, msg_counters = coordinator.dump_stats(test_folder)

    for node in nodes:
        node.dump_stats(test_folder)
    
    return full_sync_history, msg_counters


def get_config(num_nodes=10, num_iterations=100, sliding_window_size=100, d=10, error_bound=0.05,
               slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
               domain=None, domain_type=DomainType.Absolute.value, neighborhood_size=1.0, num_iterations_for_tuning=0):
    conf = dict()
    conf["num_nodes"] = num_nodes
    conf["num_iterations"] = num_iterations
    conf["num_iterations_for_tuning"] = num_iterations_for_tuning
    conf["sliding_window_size"] = sliding_window_size
    conf["d"] = d  # dimension
    conf["error_bound"] = error_bound  # additive error bound
    conf["slack_type"] = slack_type  # slack type - NoSlack (0) / Drift (1)
    conf["sync_type"] = sync_type  # sync type - Eager (0) / LazyRandom (1) / LazyLRU (2)
    conf["domain"] = domain  # 1d domain. None for (-inf, inf) or (lower, upper), which are both not None. This (lower, upper) is applied to each coordinate of x.
    conf["domain_type"] = domain_type  # domain type - Absolute (0) / Relative (1)
    conf["neighborhood_size"] = neighborhood_size  # the neighborhood size if domain_type is Relative
    return conf


def read_config_file(test_folder):
    with open(test_folder + "/config.txt") as conf_file:
        conf = conf_file.read()
    conf = eval(conf)
    return conf


def write_config_to_file(test_folder, conf):
    with open(test_folder + "/config.txt", 'w') as txt_file:
        txt_file.write(str(conf))
