import argparse
import logging
import os
import numpy as np
from automon import AutomonNode, AutomonCoordinator, SlackType, SyncType
from test_utils.data_generator import DataGeneratorQuadratic
from test_utils.functions_to_monitor import get_func_quadratic
from test_utils.stats_analysis_utils import log_num_packets_sent_and_received
from test_utils.test_utils import start_test, end_test, write_config_to_file, read_config_file
from test_utils.test_utils_zmq_sockets import run_node, run_coordinator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, dest="host", help="host", default='127.0.0.1')  # Coordinator IP
    parser.add_argument("--port", type=int, dest="port", help="port", default=6400)  # Coordinator listening port
    parser.add_argument("--node_idx", type=int, dest="node_idx", help="-1 coordinator, >= 0 node idx", default=-1)
    parser.add_argument("--error_bound", type=float, dest="error_bound", help="error bound", default=0.3)
    args = parser.parse_args()

    try:
        test_folder = start_test("distributed_quadratic", logging_level=logging.INFO)
        log_num_packets_sent_and_received(test_folder)  # Log before start

        data_folder = os.path.abspath(os.path.dirname(__file__)) + '/../datasets/quadratic/'
        conf = read_config_file(data_folder)
        conf["error_bound"] = args.error_bound
        write_config_to_file(test_folder, conf)
        H = np.loadtxt(data_folder + 'H_matrix.txt', dtype=np.float32)
        func_quadratic = get_func_quadratic(H)

        if args.node_idx == -1:
            coordinator = AutomonCoordinator(conf["num_nodes"], func_quadratic, slack_type=SlackType(conf["slack_type"]), sync_type=SyncType(conf["sync_type"]),
                                             error_bound=conf["error_bound"], neighborhood_size=conf["neighborhood_size"], domain=conf["domain"], d=conf["d"])
            run_coordinator(coordinator, args.port, conf["num_nodes"], test_folder)

        if args.node_idx >= 0:
            data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], data_file_name="data_file.txt", test_folder=data_folder, sliding_window_size=conf["sliding_window_size"])
            node = AutomonNode(args.node_idx, func_quadratic, domain=conf["domain"], d=conf["d"])
            run_node(args.host, args.port, node, args.node_idx, data_generator, test_folder)

        log_num_packets_sent_and_received(test_folder)  # Log at the end

    finally:
        end_test()
