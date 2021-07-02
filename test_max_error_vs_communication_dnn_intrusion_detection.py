from auto_mon_monitoring.node_dnn_intrusion_detection_auto_mon import NodeDnnIntrusionDetectionAutoMon
from auto_mon_monitoring.tune_neighborhood_size import tune_neighborhood_size
from coordinators.coordinator_auto_mon import CoordinatorAutoMon, DomainType
from coordinators.coordinator_rlv import CoordinatorRLV
from data_generator import DataGenerator, DataGeneratorDnnIntrusionDetection
from coordinators.coordinator_common import SlackType, SyncType
from functions_to_update_local_vector import update_local_vector_average
from jax_dnn_intrusion_detection import load_net
from rlv_monitoring.node_dnn_intrusion_detection_rlv import NodeDnnIntrusionDetectionRLV
from test_utils import start_test, end_test, run_test, get_config, write_config_to_file, read_config_file
from stats_analysis_utils import plot_figures
import logging
from object_factory import get_objects
from functions_to_monitor import set_net_params, func_dnn_intrusion_detection
from test_figures.plot_error_communication_tradeoff import plot_max_error_vs_communication


def test_error_bounds(error_bound, parent_test_folder):
    conf = read_config_file(parent_test_folder)
    data_generator = DataGenerator(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                   data_file_name="data_file.txt", test_folder=parent_test_folder,
                                   num_iterations_for_tuning=500)

    try:
        test_folder = parent_test_folder + "/threshold_" + str(error_bound)
        test_folder = start_test("error_bound_" + str(error_bound), test_folder)

        conf["error_bound"] = error_bound
        write_config_to_file(test_folder, conf)

        tuned_neighborhood_size = tune_neighborhood_size(func_dnn_intrusion_detection, NodeDnnIntrusionDetectionAutoMon,
                                                         conf, conf["k"], data_generator, conf["sliding_window_size"],
                                                         update_local_vector_average, b_check_violation_every_sample=True)
        conf["neighborhood_size"] = tuned_neighborhood_size

        logging.info("\n ###################### Start RLV test ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeDnnIntrusionDetectionRLV, CoordinatorRLV, conf, conf["k"], func_dnn_intrusion_detection)
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"],
                 update_local_vector_average, b_check_violation_every_sample=True)

        logging.info("\n ###################### Start AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeDnnIntrusionDetectionAutoMon, CoordinatorAutoMon, conf, conf["k"], func_dnn_intrusion_detection)
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"],
                 update_local_vector_average, b_check_violation_every_sample=True)

        plot_figures(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    data_folder = 'datasets/intrusion_detection/'
    net_params, net_apply = load_net(data_folder)
    set_net_params(net_params, net_apply)

    parent_test_folder = start_test("test_max_error_vs_communication_dnn_intrusion_detection")
    end_test()  # To close the logging

    conf = get_config(num_nodes=9, num_iterations=302973, sliding_window_size=20, k=41, error_bound=0.005,
                      slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                      domain_type=DomainType.Relative.value, neighborhood_size=0.5)
    write_config_to_file(parent_test_folder, conf)

    data_generator = DataGeneratorDnnIntrusionDetection(num_iterations=conf["num_iterations"],
                                                        num_nodes=conf["num_nodes"], k=conf["k"],
                                                        test_folder=parent_test_folder, num_iterations_for_tuning=500)

    error_bounds = [0.001, 0.002, 0.0027, 0.003, 0.005, 0.007, 0.01, 0.016, 0.025, 0.05, 0.5, 1.0]

    for error_bound in error_bounds:
        test_error_bounds(error_bound, parent_test_folder)

    plot_max_error_vs_communication(parent_test_folder, "DNN Intrusion Detection")
