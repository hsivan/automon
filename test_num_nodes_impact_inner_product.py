from auto_mon_monitoring.node_inner_product_auto_mon import NodeInnerProductAutoMon
from data_generator import DataGeneratorInnerProduct
from coordinators.coordinator_common import SlackType, SyncType
from coordinators.coordinator_auto_mon import CoordinatorAutoMon
from functions_to_update_local_vector import update_local_vector_average
from test_utils import start_test, end_test, run_test, get_config, write_config_to_file, read_config_file
from stats_analysis_utils import plot_figures
import logging
from object_factory import get_objects
from functions_to_monitor import func_inner_product
from test_figures.plot_num_nodes_impact import plot_num_nodes_impact_on_communication


def test_num_nodes(num_nodes, parent_test_folder):
    conf = read_config_file(parent_test_folder)

    try:
        test_folder = parent_test_folder + "/num_nodes_" + str(num_nodes)
        test_folder = start_test("num_nodes_" + str(num_nodes), test_folder)

        conf["num_nodes"] = num_nodes
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], k=conf["k"], test_folder=parent_test_folder)

        logging.info("\n ###################### Start AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeInnerProductAutoMon, CoordinatorAutoMon, conf, 2 * conf["k"], func_inner_product)
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_average)

        plot_figures(test_folder)

    finally:
        end_test()


if __name__ == "__main__":
    parent_test_folder = start_test("test_num_nodes_impact_inner_product")
    end_test()  # To close the logging

    conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, k=20, error_bound=0.3,
                      slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
    write_config_to_file(parent_test_folder, conf)

    num_nodes_arr = [10, 20, 40, 60, 100, 500]

    for num_nodes in num_nodes_arr:
        test_num_nodes(num_nodes, parent_test_folder)

    plot_num_nodes_impact_on_communication(parent_test_folder)
