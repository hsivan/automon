from automon.automon.node_automon import NodeAutoMon
from test_utils.functions_to_monitor import func_kld
from test_utils.tune_neighborhood_size import tune_neighborhood_size
from automon.automon.coordinator_automon import CoordinatorAutoMon
from test_utils.data_generator import DataGeneratorKldAirQuality
from test_utils.test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging
from test_utils.object_factory import get_objects
from experiments.visualization.plot_error_communication_tradeoff import plot_max_error_vs_communication


def test_error_bounds(error_bound, parent_test_folder):
    conf = read_config_file(parent_test_folder)
    data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                data_file_name="data_file.txt", test_folder=parent_test_folder, d=conf["d"],
                                                num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

    try:
        test_folder = parent_test_folder + "/threshold_" + str(error_bound)
        test_folder = start_test("error_bound_" + str(error_bound), test_folder)

        conf["error_bound"] = error_bound
        write_config_to_file(test_folder, conf)

        logging.info("\n###################### Start AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeAutoMon, CoordinatorAutoMon, conf, func_kld)
        tune_neighborhood_size(coordinator, nodes, conf, data_generator)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    parent_test_folder = start_test("test_max_error_vs_communication_kld_air_quality")
    end_test()  # To close the logging

    data_folder = '../datasets/air_quality/'
    conf = read_config_file(data_folder)
    write_config_to_file(parent_test_folder, conf)
    data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                d=conf["d"], test_folder=parent_test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"],
                                                sliding_window_size=conf["sliding_window_size"])

    error_bounds = [0.003, 0.004, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]

    for error_bound in error_bounds:
        test_error_bounds(error_bound, parent_test_folder)

    plot_max_error_vs_communication(parent_test_folder, "KLD")
