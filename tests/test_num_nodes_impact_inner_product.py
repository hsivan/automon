from automon.automon.nodes_automon import NodeInnerProductAutoMon
from automon.data_generator import DataGeneratorInnerProduct
from automon.coordinator_common import SlackType, SyncType
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.test_utils import start_test, end_test, run_test, get_config, write_config_to_file, read_config_file
from automon.stats_analysis_utils import plot_monitoring_stats
import logging
from automon.object_factory import get_objects
from tests.visualization.plot_num_nodes_impact import plot_num_nodes_impact_on_communication


def test_num_nodes(num_nodes, parent_test_folder):
    conf = read_config_file(parent_test_folder)

    try:
        test_folder = parent_test_folder + "/num_nodes_" + str(num_nodes)
        test_folder = start_test("num_nodes_" + str(num_nodes), test_folder)

        conf["num_nodes"] = num_nodes
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder)

        logging.info("\n###################### Start AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeInnerProductAutoMon, CoordinatorAutoMon, conf)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        plot_monitoring_stats(test_folder)

    finally:
        end_test()


if __name__ == "__main__":
    parent_test_folder = start_test("test_num_nodes_impact_inner_product")
    end_test()  # To close the logging

    conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, d=40, error_bound=0.3,
                      slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
    write_config_to_file(parent_test_folder, conf)

    num_nodes_arr = [10, 20, 40, 60, 100, 500]

    for num_nodes in num_nodes_arr:
        test_num_nodes(num_nodes, parent_test_folder)

    plot_num_nodes_impact_on_communication(parent_test_folder)