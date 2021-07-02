from auto_mon_monitoring.node_kld_auto_mon import NodeKLDAutoMon
from auto_mon_monitoring.tune_neighborhood_size import tune_neighborhood_size
from coordinators.coordinator_auto_mon import CoordinatorAutoMon, DomainType
from data_generator import DataGenerator, DataGeneratorKldAirQuality
from coordinators.coordinator_common import SlackType, SyncType
from functions_to_update_local_vector import update_local_vector_concatenated_frequency_vectors
from test_utils import start_test, end_test, run_test, get_config, write_config_to_file, read_config_file
from stats_analysis_utils import plot_figures
import logging
from object_factory import get_objects
from functions_to_monitor import func_kld
from test_figures.plot_error_communication_tradeoff import plot_max_error_vs_communication


def test_error_bounds(error_bound, parent_test_folder):
    conf = read_config_file(parent_test_folder)
    data_generator = DataGenerator(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                   data_file_name="data_file.txt", test_folder=parent_test_folder,
                                   num_iterations_for_tuning=300)

    try:
        test_folder = parent_test_folder + "/threshold_" + str(error_bound)
        test_folder = start_test("error_bound_" + str(error_bound), test_folder)

        conf["error_bound"] = error_bound
        write_config_to_file(test_folder, conf)

        tuned_neighborhood_size = tune_neighborhood_size(func_kld, NodeKLDAutoMon, conf, 2 * conf["k"], data_generator, conf["sliding_window_size"], update_local_vector_concatenated_frequency_vectors)
        conf["neighborhood_size"] = tuned_neighborhood_size

        logging.info("\n ###################### Start AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeKLDAutoMon, CoordinatorAutoMon, conf, 2*conf["k"], func_kld)
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_concatenated_frequency_vectors)

        plot_figures(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    parent_test_folder = start_test("test_max_error_vs_communication_kld_air_quality")
    end_test()  # To close the logging

    conf = get_config(num_nodes=12, num_iterations=30000, sliding_window_size=200, k=10,
                      slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, domain=(0, 1),
                      domain_type=DomainType.Relative.value)
    write_config_to_file(parent_test_folder, conf)

    data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                k=conf["k"], test_folder=parent_test_folder, num_iterations_for_tuning=300)

    error_bounds = [0.001, 0.003, 0.004, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]

    for error_bound in error_bounds:
        test_error_bounds(error_bound, parent_test_folder)

    plot_max_error_vs_communication(parent_test_folder, "kld")
