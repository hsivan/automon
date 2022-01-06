import argparse
import os
from automon.auto_mon.coordinator_automon import CoordinatorAutoMon
from automon.auto_mon.node_automon import NodeAutoMon
from test_utils.data_generator import DataGeneratorDnnIntrusionDetection
from test_utils.functions_to_monitor import set_net_params, func_dnn_intrusion_detection
from test_utils.jax_dnn_intrusion_detection import load_net
from test_utils.stats_analysis_utils import log_num_packets_sent_and_received
from test_utils.test_utils import start_test, end_test, write_config_to_file, read_config_file
from test_utils.object_factory import get_node, get_coordinator
from test_utils.test_utils_zmq_sockets import run_coordinator, run_node

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, dest="host", help="host", default='127.0.0.1')  # Coordinator IP
    parser.add_argument("--port", type=int, dest="port", help="port", default=6400)  # Coordinator listening port
    parser.add_argument("--node_idx", type=int, dest="node_idx", help="-1 coordinator, >= 0 node idx", default=-1)
    parser.add_argument("--error_bound", type=float, dest="error_bound", help="error bound", default=0.003)
    args = parser.parse_args()

    # Theses values were taken from the results of test_max_error_vs_communication_dnn_intrusion_detection.py
    error_bound_to_neighborhood_size = {0.001: 0.010416666666666666, 0.002: 0.017361111111111112, 0.0027: 0.027777777777777776,
                                        0.003: 0.027777777777777776, 0.005: 0.03125, 0.007: 0.03125, 0.01: 0.041666666666666664,
                                        0.016: 0.041666666666666664, 0.025: 0.041666666666666664, 0.05: 0.041666666666666664}

    try:
        test_folder = start_test("distributed_dnn_intrusion_detection")
        log_num_packets_sent_and_received(test_folder)  # Log before start

        data_folder = os.path.abspath(os.path.abspath(os.path.dirname(__file__))) + '/../datasets/intrusion_detection/'
        conf = read_config_file(data_folder)
        conf["error_bound"] = args.error_bound
        conf["neighborhood_size"] = error_bound_to_neighborhood_size[args.error_bound]
        write_config_to_file(test_folder, conf)
        net_params, net_apply = load_net(data_folder)
        set_net_params(net_params, net_apply)

        if args.node_idx == -1:
            coordinator = get_coordinator(CoordinatorAutoMon, NodeAutoMon, conf, func_dnn_intrusion_detection)
            run_coordinator(coordinator, args.port, conf["num_nodes"], test_folder)

        if args.node_idx >= 0:
            data_generator = DataGeneratorDnnIntrusionDetection(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])
            node = get_node(NodeAutoMon, conf["domain"], conf["d"], args.node_idx, func_dnn_intrusion_detection)
            run_node(args.host, args.port, node, args.node_idx, data_generator, test_folder, b_single_sample_per_round=True)

        log_num_packets_sent_and_received(test_folder)  # Log at the end

    finally:
        end_test()
