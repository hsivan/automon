import argparse
import os
from automon.auto_mon.coordinator_automon import CoordinatorAutoMon
from automon.auto_mon.node_automon import NodeAutoMon
from test_utils.data_generator import DataGeneratorKldAirQuality
from test_utils.functions_to_monitor import func_kld
from test_utils.stats_analysis_utils import log_num_packets_sent_and_received
from test_utils.test_utils import start_test, end_test, write_config_to_file, read_config_file
from test_utils.object_factory import get_node, get_coordinator
from test_utils.test_utils_zmq_sockets import run_coordinator, run_node

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, dest="host", help="host", default='127.0.0.1')  # Coordinator IP
    parser.add_argument("--port", type=int, dest="port", help="port", default=6400)  # Coordinator listening port
    parser.add_argument("--node_idx", type=int, dest="node_idx", help="-1 coordinator, >= 0 node idx", default=-1)
    parser.add_argument("--error_bound", type=float, dest="error_bound", help="error bound", default=0.1)
    args = parser.parse_args()

    # Theses values were taken from the results of test_max_error_vs_communication_kld_air_quality.py
    error_bound_to_neighborhood_size = {0.003: 0.059895833333333336, 0.004: 0.0859375, 0.005: 0.08854166666666667, 0.01: 0.10069444444444445,
                                        0.02: 0.1527777777777778, 0.04: 0.25, 0.06: 0.2361111111111111, 0.08: 0.2361111111111111,
                                        0.1: 0.2361111111111111, 0.12: 0.2361111111111111, 0.14: 0.2361111111111111}

    try:
        test_folder = start_test("distributed_kld_air_quality")
        log_num_packets_sent_and_received(test_folder)  # Log before start

        data_folder = os.path.abspath(os.path.abspath(os.path.dirname(__file__))) + '/../datasets/air_quality/'
        conf = read_config_file(data_folder)
        conf["error_bound"] = args.error_bound
        conf["neighborhood_size"] = error_bound_to_neighborhood_size[args.error_bound]
        write_config_to_file(test_folder, conf)

        if args.node_idx == -1:
            coordinator = get_coordinator(CoordinatorAutoMon, NodeAutoMon, conf, func_kld)
            run_coordinator(coordinator, args.port, conf["num_nodes"], test_folder)

        if args.node_idx >= 0:
            data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])
            node = get_node(NodeAutoMon, conf["domain"], conf["d"], args.node_idx, func_kld)
            run_node(args.host, args.port, node, args.node_idx, data_generator, test_folder)

        log_num_packets_sent_and_received(test_folder)  # Log at the end

    finally:
        end_test()
