import argparse
import os
import numpy as np
from automon.automon.nodes_automon import NodeQuadraticAutoMon
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.data_generator import DataGeneratorQuadratic
from automon.functions_to_monitor import set_H
from automon.stats_analysis_utils import log_num_packets_sent_and_received
from automon.test_utils import start_test, end_test, write_config_to_file, read_config_file
from automon.object_factory import get_node, get_coordinator
from automon.test_utils_zmq_sockets import run_node, run_coordinator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, dest="host", help="host", default='127.0.0.1')  # Coordinator IP
    parser.add_argument("--port", type=int, dest="port", help="port", default=6400)  # Coordinator listening port
    parser.add_argument("--node_idx", type=int, dest="node_idx", help="-1 coordinator, >= 0 node idx", default=-1)
    parser.add_argument("--error_bound", type=float, dest="error_bound", help="error bound", default=0.3)
    args = parser.parse_args()

    try:
        test_folder = start_test("distributed_quadratic")
        log_num_packets_sent_and_received(test_folder)  # Log before start

        data_folder = os.path.abspath(os.path.dirname(__file__)) + '/../datasets/quadratic/'
        conf = read_config_file(data_folder)
        conf["error_bound"] = args.error_bound
        write_config_to_file(test_folder, conf)
        H = np.loadtxt(data_folder + 'H_matrix.txt', dtype=np.float32)
        set_H(conf["d"], H)

        if args.node_idx == -1:
            coordinator = get_coordinator(CoordinatorAutoMon, NodeQuadraticAutoMon, conf)
            run_coordinator(coordinator, args.port, conf["num_nodes"], test_folder)

        if args.node_idx >= 0:
            data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], data_file_name="data_file.txt", test_folder=data_folder)
            node = get_node(NodeQuadraticAutoMon, conf["domain"], conf["d"], args.node_idx)
            run_node(args.host, args.port, node, args.node_idx, data_generator, conf["num_nodes"], conf["sliding_window_size"], test_folder)

        log_num_packets_sent_and_received(test_folder)  # Log at the end

    finally:
        end_test()