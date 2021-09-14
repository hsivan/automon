from automon.automon.nodes_automon import NodeKLDAutoMon
from automon.automon.tune_neighborhood_size import tune_neighborhood_size
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.rlv.coordinator_rlv import CoordinatorRLV
from automon.data_generator import DataGeneratorKldAirQuality
from automon.rlv.nodes_rlv import NodeKLDRLV
from automon.test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from automon.stats_analysis_utils import plot_monitoring_stats
import logging
from automon.object_factory import get_objects

if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_kld_air_quality")

        '''conf = get_config(num_nodes=12, num_iterations=30000, sliding_window_size=200, d=20, error_bound=0.1,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, domain=(0, 1),
                          domain_type=DomainType.Relative.value, num_iterations_for_tuning=300)'''
        data_folder = '../datasets/air_quality/'
        conf = read_config_file(data_folder)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                    d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"])

        logging.info("\n###################### Start KLD RLV test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeKLDRLV, CoordinatorRLV, conf)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        logging.info("\n###################### Start KLD AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeKLDAutoMon, CoordinatorAutoMon, conf)
        tune_neighborhood_size(coordinator, nodes, conf, data_generator)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
