from automon.automon.node_common_automon import NodeCommonAutoMon
from test_utils.tune_neighborhood_size import tune_neighborhood_size
from automon.automon.coordinator_automon import CoordinatorAutoMon
from test_utils.data_generator import DataGeneratorDnnIntrusionDetection
from test_utils.jax_dnn_intrusion_detection import load_net
from test_utils.test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging
from test_utils.object_factory import get_objects
from test_utils.functions_to_monitor import set_net_params, func_dnn_intrusion_detection
from experiments.visualization.plot_error_communication_tradeoff import plot_max_error_vs_communication


def test_error_bounds(error_bound, parent_test_folder):
    conf = read_config_file(parent_test_folder)
    data_generator = DataGeneratorDnnIntrusionDetection(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                        data_file_name="data_file.txt", test_folder=parent_test_folder, d=conf["d"],
                                                        num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

    try:
        test_folder = parent_test_folder + "/threshold_" + str(error_bound)
        test_folder = start_test("error_bound_" + str(error_bound), test_folder)

        conf["error_bound"] = error_bound
        write_config_to_file(test_folder, conf)

        logging.info("\n###################### Start AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeCommonAutoMon, CoordinatorAutoMon, conf, func_dnn_intrusion_detection)
        tune_neighborhood_size(coordinator, nodes, conf, data_generator, b_single_sample_per_round=True)
        run_test(data_generator, coordinator, nodes, test_folder, b_single_sample_per_round=True)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    data_folder = '../datasets/intrusion_detection/'
    net_params, net_apply = load_net(data_folder)
    set_net_params(net_params, net_apply)

    parent_test_folder = start_test("test_max_error_vs_communication_dnn_intrusion_detection")
    end_test()  # To close the logging

    data_folder = '../datasets/intrusion_detection/'
    conf = read_config_file(data_folder)
    write_config_to_file(parent_test_folder, conf)
    data_generator = DataGeneratorDnnIntrusionDetection(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                        d=conf["d"], test_folder=parent_test_folder,
                                                        num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

    error_bounds = [0.001, 0.002, 0.0027, 0.003, 0.005, 0.007, 0.01, 0.016, 0.025, 0.05]

    for error_bound in error_bounds:
        test_error_bounds(error_bound, parent_test_folder)

    plot_max_error_vs_communication(parent_test_folder, "DNN Intrusion Detection")
