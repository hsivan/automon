from automon import SlackType, SyncType
import os
import datetime
from timeit import default_timer as timer
import logging
import sys

logger = logging.getLogger('automon')


def _prepare_test_folder(test_name, test_folder=""):
    if test_folder == "":
        test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        test_folder = os.path.join(os.getcwd(), 'test_results/results_' + test_name + "_" + test_timestamp)
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s')
    output_file_handler = logging.FileHandler(test_folder + "/" + test_name + '.log', mode='w')
    output_file_handler.setLevel(logging.INFO)  # Can change to DEBUG to see send and received message logs from messages_common
    output_file_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.WARNING)  # Can change to DEBUG/INFO to see all log messages
    stdout_handler.setFormatter(formatter)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    logger.propagate = False

    logging.getLogger('matplotlib.backends.backend_pdf').setLevel(logging.WARNING)
    logger.info('Started test - ' + test_name)
    return test_folder


def start_test(test_name, test_folder=""):
    test_folder = _prepare_test_folder(test_name, test_folder)
    return test_folder


def end_test():
    logger.info('Finished test')

    handlers = logger.handlers
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def _from_violation_to_sync(coordinator, nodes, message_violations):
    messages_from_node_to_coordinator = message_violations
    while len(messages_from_node_to_coordinator) > 0:  # There are no more messages from the nodes after the coordinator sends sync
        messages_from_coordinator_to_nodes = coordinator.parse_message(messages_from_node_to_coordinator)
        messages_from_node_to_coordinator = b''
        for node_idx, message in messages_from_coordinator_to_nodes:
            message_from_node_to_coordinator = nodes[node_idx].parse_message(message)
            if message_from_node_to_coordinator is not None:
                messages_from_node_to_coordinator += message_from_node_to_coordinator


def _test_data_loop(coordinator, nodes, data_generator, b_single_sample_per_round):
    # First data round after all sliding windows are full: provide the verifier the global vector, provide all the
    # nodes their local vectors, and call the coordinator to resolve violations.
    global_vector = data_generator.get_global_vector()
    coordinator.verifier.set_new_data_point(global_vector)  # Update the verifier
    message_violations = b''
    for node_idx, node in enumerate(nodes):
        local_vector = data_generator.get_local_vector(node_idx)
        message_violation = node.update_data(local_vector)
        if message_violation is not None and len(message_violation) > 0:
            message_violations += message_violation
    if len(message_violations) > 0:
        _from_violation_to_sync(coordinator, nodes, message_violations)
    coordinator.update_statistics()

    # For the rest of the data rounds: read data from stream, update verifier global vector and nodes local
    # vectors, and call the coordinator to resolve violations.
    message_violations = b''
    num_samples = data_generator.get_num_samples_left()
    prev = timer()
    for i in range(num_samples):
        now = timer()
        # Print progress message every 5 minutes
        if now - prev > 5 * 60:
            print("Completed " + str(i/num_samples) + "%")
            sys.stdout.flush()
            prev = now
        local_vector, node_idx = data_generator.get_next_data_point()
        global_vector = data_generator.get_global_vector()

        # Update the verifier first with the global vector, so the coordinator can check, after the update
        # of the node, if the violation is "false local", "false global" or "true".
        coordinator.verifier.set_new_data_point(global_vector)  # Update the verifier
        message_violation = nodes[node_idx].update_data(local_vector)
        if message_violation is not None and len(message_violation) > 0:
            message_violations += message_violation

        if b_single_sample_per_round:
            if len(message_violations) > 0:
                _from_violation_to_sync(coordinator, nodes, message_violations)
            coordinator.update_statistics()
            message_violations = b''
        else:
            if i % len(nodes) == len(nodes) - 1:
                if len(message_violations) > 0:
                    _from_violation_to_sync(coordinator, nodes, message_violations)
                coordinator.update_statistics()
                message_violations = b''
        i += 1


def run_test(data_generator, coordinator, nodes, test_folder, b_single_sample_per_round=False):
    logger.info("num_nodes " + str(len(nodes)) + ", num_iterations " + str(data_generator.get_num_iterations()) + ", data_generator state " + str(data_generator.state))
    coordinator.b_simulation = True  # This is a simulation of distributed system on a single machine with 'experiment manager' that simulates the system in a serialized manner.

    # Send data to nodes
    start = timer()
    _test_data_loop(coordinator, nodes, data_generator, b_single_sample_per_round)
    end = timer()
    logger.info("The test took: " + str(end - start) + " seconds")
    full_sync_history, msg_counters = coordinator.dump_stats(test_folder)

    for node in nodes:
        node.dump_stats(test_folder)
    
    return full_sync_history, msg_counters


def get_config(num_nodes=10, num_iterations=100, sliding_window_size=100, d=10, error_bound=0.05,
               slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
               domain=None, neighborhood_size=None, num_iterations_for_tuning=0):
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
    conf["neighborhood_size"] = neighborhood_size  # the neighborhood size, or None if the neighborhood is the entire domain
    return conf


def read_config_file(test_folder):
    with open(test_folder + "/config.txt") as conf_file:
        conf = conf_file.read()
    conf = eval(conf)
    return conf


def write_config_to_file(test_folder, conf):
    with open(test_folder + "/config.txt", 'w') as txt_file:
        txt_file.write(str(conf))
