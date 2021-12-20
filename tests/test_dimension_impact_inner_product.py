from utils.nodes_automon import NodeInnerProductAutoMon
from automon.automon.coordinator_automon import CoordinatorAutoMon
from utils.data_generator import DataGeneratorInnerProduct
from automon.coordinator_common import SlackType, SyncType
from utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from utils.stats_analysis_utils import plot_monitoring_stats
import logging
from utils.object_factory import get_objects


def test_dimension(dimension, parent_test_folder):
    assert(dimension % 2 == 0)

    try:
        test_folder = parent_test_folder + "/dimension_" + str(dimension)
        test_folder = start_test("dimension_" + str(dimension), test_folder)

        conf = get_config(num_nodes=12, num_iterations=1020, sliding_window_size=20, d=dimension, error_bound=0.25,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                   d=conf["d"], test_folder=test_folder, sliding_window_size=conf["sliding_window_size"])

        logging.info("\n###################### Start inner product AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeInnerProductAutoMon, CoordinatorAutoMon, conf)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    parent_test_folder = start_test("test_dimension_impact_inner_product")
    end_test()  # To close the logging

    dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]

    for dimension in dimensions:
        test_dimension(dimension, parent_test_folder)
