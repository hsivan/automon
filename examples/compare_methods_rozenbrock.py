from automon.automon.node_automon import NodeAutoMon
from automon.rlv.node_rlv import NodeRLV
from test_utils.functions_to_monitor import func_rozenbrock
from test_utils.tune_neighborhood_size import tune_neighborhood_size
from automon.rlv.coordinator_rlv import CoordinatorRLV
from test_utils.data_generator import DataGeneratorRozenbrock
from automon.coordinator_common import SlackType, SyncType
from automon.automon.coordinator_automon import CoordinatorAutoMon
from test_utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging
from test_utils.object_factory import get_objects

if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_rozenbrock")

        conf = get_config(num_nodes=10, num_iterations=500, sliding_window_size=20, d=2, error_bound=0.1,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          neighborhood_size=0.15, num_iterations_for_tuning=200)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorRozenbrock(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                 d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

        logging.info("\n###################### Start Rozenbrock RLV test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeRLV, CoordinatorRLV, conf, func_rozenbrock)
        run_test(data_generator, coordinator, nodes, test_folder)

        logging.info("\n###################### Start Rozenbrock AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeAutoMon, CoordinatorAutoMon, conf, func_rozenbrock)
        tune_neighborhood_size(coordinator, nodes, conf, data_generator)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
