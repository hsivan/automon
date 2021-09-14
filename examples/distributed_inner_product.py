import argparse
import os
from automon.automon.nodes_automon import NodeInnerProductAutoMon
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.data_generator import DataGeneratorInnerProduct
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
        test_folder = start_test("distributed_inner_product")
        log_num_packets_sent_and_received(test_folder)  # Log before start

        data_folder = os.path.abspath(os.path.dirname(__file__)) + "/../datasets/inner_product/"
        conf = read_config_file(data_folder)
        conf["error_bound"] = args.error_bound
        write_config_to_file(test_folder, conf)

        if args.node_idx == -1:
            coordinator = get_coordinator(CoordinatorAutoMon, NodeInnerProductAutoMon, conf)
            run_coordinator(coordinator, args.port, conf["num_nodes"], test_folder)

        if args.node_idx >= 0:
            data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name="data_file.txt", d=conf["d"], test_folder=data_folder)
            node = get_node(NodeInnerProductAutoMon, conf["domain"], conf["d"], args.node_idx)
            run_node(args.host, args.port, node, args.node_idx, data_generator, conf["num_nodes"], conf["sliding_window_size"], test_folder)

        log_num_packets_sent_and_received(test_folder)  # Log at the end

    finally:
        end_test()
