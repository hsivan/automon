from automon.automon.nodes_automon import NodeVarianceAutoMon
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.gm.coordinator_gm import CoordinatorGM
from automon.data_generator import DataGeneratorVariance
from automon.coordinator_common import SlackType, SyncType
from automon.gm.node_variance_gm import NodeVarianceGM
from automon.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from automon.stats_analysis_utils import plot_monitoring_stats
import logging
from automon.object_factory import get_objects

if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_variance")

        conf = get_config(num_nodes=10, num_iterations=500, sliding_window_size=5, d=2, error_bound=1,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorVariance(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], test_folder=test_folder)

        logging.info("\n###################### Start variance GM test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeVarianceGM, CoordinatorGM, conf)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        logging.info("\n###################### Start variance AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeVarianceAutoMon, CoordinatorAutoMon, conf)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
