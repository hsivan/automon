from automon.automon.nodes_automon import NodeRozenbrockAutoMon
from automon.automon.tune_neighborhood_size import tune_neighborhood_size
from automon.rlv.coordinator_rlv import CoordinatorRLV
from automon.data_generator import DataGeneratorRozenbrock
from automon.coordinator_common import SlackType, SyncType
from automon.automon.coordinator_automon import CoordinatorAutoMon, DomainType
from automon.rlv.nodes_rlv import NodeRozenbrockRLV
from automon.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from automon.stats_analysis_utils import plot_monitoring_stats
import logging
from automon.object_factory import get_objects

if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_rozenbrock")

        conf = get_config(num_nodes=10, num_iterations=500, sliding_window_size=20, d=2, error_bound=0.1,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          domain_type=DomainType.Relative.value, neighborhood_size=0.15, num_iterations_for_tuning=200)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorRozenbrock(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                 d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"])

        logging.info("\n###################### Start Rozenbrock RLV test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeRozenbrockRLV, CoordinatorRLV, conf)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        logging.info("\n###################### Start Rozenbrock AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeRozenbrockAutoMon, CoordinatorAutoMon, conf)
        tune_neighborhood_size(coordinator, nodes, conf, data_generator)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
