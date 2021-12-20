from utils.nodes_automon import NodeEntropyAutoMon
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.gm.coordinator_gm import CoordinatorGM
from utils.data_generator import DataGeneratorEntropy
from automon.coordinator_common import SlackType, SyncType
from automon.gm.node_entropy_gm import NodeEntropyGM
from utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from utils.stats_analysis_utils import plot_monitoring_stats
import logging
from utils.object_factory import get_objects

if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_entropy")

        conf = get_config(num_nodes=10, num_iterations=1100, sliding_window_size=100, d=10, error_bound=0.05,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, domain=(0, 1))
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorEntropy(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                              d=conf["d"], test_folder=test_folder, sliding_window_size=conf["sliding_window_size"])

        logging.info("\n###################### Start entropy GM test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeEntropyGM, CoordinatorGM, conf)
        run_test(data_generator, coordinator, nodes, test_folder)

        logging.info("\n###################### Start entropy AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeEntropyAutoMon, CoordinatorAutoMon, conf)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
