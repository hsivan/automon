from automon.automon.nodes_automon import NodeInnerProductAutoMon
from automon.cb.node_inner_product_cb import NodeInnerProductCB
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.cb.coordinator_cb import CoordinatorCB
from automon.data_generator import DataGeneratorInnerProduct
from automon.coordinator_common import SlackType, SyncType
from automon.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from automon.stats_analysis_utils import plot_monitoring_stats
import logging
from automon.object_factory import get_objects

if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_inner_product")

        conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, d=40, error_bound=0.3,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                   d=conf["d"], test_folder=test_folder)

        logging.info("\n###################### Start inner product CB test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeInnerProductCB, CoordinatorCB, conf)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        logging.info("\n###################### Start inner product AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeInnerProductAutoMon, CoordinatorAutoMon, conf)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        plot_monitoring_stats(test_folder)

    finally:
        end_test()