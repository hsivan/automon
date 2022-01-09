import argparse
import os
from automon.automon.automon_coordinator import AutomonCoordinator
from automon.automon.automon_node import AutomonNode
from test_utils.data_generator import DataGeneratorInnerProduct
from test_utils.functions_to_monitor import func_inner_product
from test_utils.stats_analysis_utils import log_num_packets_sent_and_received
from test_utils.test_utils import start_test, end_test, write_config_to_file, read_config_file
from test_utils.object_factory import get_node, get_coordinator
from test_utils.test_utils_zmq_sockets import run_node, run_coordinator


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
            coordinator = get_coordinator(AutomonCoordinator, AutomonNode, conf, func_inner_product)
            run_coordinator(coordinator, args.port, conf["num_nodes"], test_folder)

        if args.node_idx >= 0:
            data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name="data_file.txt", d=conf["d"], test_folder=data_folder, sliding_window_size=conf["sliding_window_size"])
            node = get_node(AutomonNode, conf["domain"], conf["d"], args.node_idx, func_inner_product)
            run_node(args.host, args.port, node, args.node_idx, data_generator, test_folder)

        log_num_packets_sent_and_received(test_folder)  # Log at the end

    finally:
        end_test()
