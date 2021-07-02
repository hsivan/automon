from auto_mon_monitoring.node_inner_product_auto_mon import NodeInnerProductAutoMon
from coordinators.coordinator_auto_mon import CoordinatorAutoMon
from data_generator import DataGeneratorInnerProduct
from coordinators.coordinator_common import SlackType, SyncType
from functions_to_update_local_vector import update_local_vector_average
from test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from stats_analysis_utils import plot_figures
import logging
from object_factory import get_objects
from functions_to_monitor import func_inner_product


def test_dimension(dimension, parent_test_folder):
    assert(dimension % 2 == 0)
    k = dimension // 2

    try:
        test_folder = parent_test_folder + "/dimension_" + str(dimension)
        test_folder = start_test("dimension_" + str(dimension), test_folder)

        conf = get_config(num_nodes=12, num_iterations=1020, sliding_window_size=20, k=k, error_bound=0.25,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                   k=conf["k"], test_folder=test_folder)

        logging.info("\n ###################### Start inner produce AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeInnerProductAutoMon, CoordinatorAutoMon, conf, 2 * conf["k"], func_inner_product)
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_average)

        plot_figures(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    parent_test_folder = start_test("test_dimension_impact_inner_product")
    end_test()  # To close the logging

    dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]

    for dimension in dimensions:
        test_dimension(dimension, parent_test_folder)
