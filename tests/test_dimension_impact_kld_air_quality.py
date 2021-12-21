from automon.automon.node_common_automon import NodeCommonAutoMon
from automon_utils.functions_to_monitor import func_kld
from automon_utils.tune_neighborhood_size import tune_neighborhood_size
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon_utils.data_generator import DataGeneratorKldAirQuality
from automon_utils.test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from automon_utils.stats_analysis_utils import plot_monitoring_stats
import logging
from automon_utils.object_factory import get_objects


def test_dimension(dimension, parent_test_folder):
    assert(dimension % 2 == 0)

    try:
        test_folder = parent_test_folder + "/dimension_" + str(dimension)
        test_folder = start_test("dimension_" + str(dimension), test_folder)

        data_folder = '../datasets/air_quality/'
        conf = read_config_file(data_folder)
        conf["num_iterations"] = 1200
        conf["d"] = dimension
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                    d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

        logging.info("\n###################### Start KLD AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeCommonAutoMon, CoordinatorAutoMon, conf, func_kld)
        tune_neighborhood_size(coordinator, nodes, conf, data_generator)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    parent_test_folder = start_test("test_dimension_impact_kld_air_quality")
    end_test()  # To close the logging

    dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]

    for dimension in dimensions:
        test_dimension(dimension, parent_test_folder)
