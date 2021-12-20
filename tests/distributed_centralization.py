import argparse
import os
import time

import numpy as np
from utils.data_generator import DataGeneratorQuadratic, DataGeneratorKldAirQuality, DataGeneratorDnnIntrusionDetection, DataGeneratorInnerProduct
from utils.functions_to_monitor import set_H, set_net_params
from utils.jax_dnn_intrusion_detection import load_net
from utils.stats_analysis_utils import log_num_packets_sent_and_received
from utils.test_utils import start_test, end_test, write_config_to_file, read_config_file
from tests.test_utils_zmq_sockets_centralization import run_centralization_node, run_dummy_coordinator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, dest="host", help="host", default='127.0.0.1')  # Coordinator IP
    parser.add_argument("--port", type=int, dest="port", help="port", default=6400)  # Coordinator listening port
    parser.add_argument("--node_idx", type=int, dest="node_idx", help="-1 coordinator, >= 0 node idx", default=-1)
    parser.add_argument("--type", type=str, dest="type", help="experiment type (inner_product/quadratic/kld/dnn)", default="inner_product")
    args = parser.parse_args()

    try:
        test_folder = start_test("distributed_" + args.type + "_centralization")
        log_num_packets_sent_and_received(test_folder)  # Log before start

        if args.type == "quadratic":
            data_folder = os.path.abspath(os.path.dirname(__file__)) + '/../datasets/quadratic/'
            conf = read_config_file(data_folder)
            H = np.loadtxt(data_folder + 'H_matrix.txt', dtype=np.float32)
            set_H(conf["d"], H)
            data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], data_file_name="data_file.txt", test_folder=data_folder, sliding_window_size=conf["sliding_window_size"])
        if args.type == "inner_product":
            data_folder = os.path.abspath(os.path.dirname(__file__)) + "/../datasets/inner_product/"
            conf = read_config_file(data_folder)
            data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name="data_file.txt", d=conf["d"], test_folder=data_folder, sliding_window_size=conf["sliding_window_size"])
        if args.type == "kld":
            data_folder = os.path.abspath(os.path.abspath(os.path.dirname(__file__))) + '/../datasets/air_quality/'
            conf = read_config_file(data_folder)
            data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])
        if args.type == "dnn":
            data_folder = os.path.abspath(os.path.abspath(os.path.dirname(__file__))) + '/../datasets/intrusion_detection/'
            conf = read_config_file(data_folder)
            write_config_to_file(test_folder, conf)
            net_params, net_apply = load_net(data_folder)
            set_net_params(net_params, net_apply)
            data_generator = DataGeneratorDnnIntrusionDetection(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])
        write_config_to_file(test_folder, conf)

        if args.node_idx == -1:
            run_dummy_coordinator(args.host, args.port, conf["num_nodes"])

        if args.node_idx >= 0:
            run_centralization_node(args.host, args.port, args.node_idx, data_generator)

        log_num_packets_sent_and_received(test_folder)  # Log at the end

    finally:
        end_test()
        time.sleep(10)  # Allow Nethogs to record the process traffic
