from automon.automon.automon_node import AutomonNode
from test_utils.functions_to_monitor import func_inner_product
from automon.cb.cb_inner_product_node import CbInnerProductNode
from automon.automon.automon_coordinator import AutomonCoordinator
from automon.cb.cb_coordinator import CbCoordinator
from test_utils.data_generator import DataGeneratorInnerProduct
from automon.common_coordinator import SlackType, SyncType
from test_utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging
from test_utils.object_factory import get_objects

if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_inner_product")

        conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, d=40, error_bound=0.3,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                   d=conf["d"], test_folder=test_folder, sliding_window_size=conf["sliding_window_size"])

        logging.info("\n###################### Start inner product CB test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(CbInnerProductNode, CbCoordinator, conf, func_inner_product)
        run_test(data_generator, coordinator, nodes, test_folder)

        logging.info("\n###################### Start inner product AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(AutomonNode, AutomonCoordinator, conf, func_inner_product)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
