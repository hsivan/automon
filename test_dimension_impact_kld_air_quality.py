from auto_mon_monitoring.node_kld_auto_mon import NodeKLDAutoMon
from auto_mon_monitoring.tune_neighborhood_size import tune_neighborhood_size
from coordinators.coordinator_auto_mon import CoordinatorAutoMon, DomainType
from data_generator import DataGeneratorKldAirQuality
from coordinators.coordinator_common import SlackType, SyncType
from functions_to_update_local_vector import update_local_vector_concatenated_frequency_vectors
from test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from stats_analysis_utils import plot_figures
import logging
from object_factory import get_objects
from functions_to_monitor import func_kld


def test_dimension(dimension, parent_test_folder):
    assert(dimension % 2 == 0)
    k = dimension // 2

    try:
        test_folder = parent_test_folder + "/dimension_" + str(dimension)
        test_folder = start_test("dimension_" + str(dimension), test_folder)

        conf = get_config(num_nodes=12, num_iterations=1200, sliding_window_size=200, k=k, error_bound=0.1,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, domain=(0, 1),
                          domain_type=DomainType.Relative.value)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                    k=conf["k"], test_folder=test_folder, num_iterations_for_tuning=300)

        tuned_neighborhood_size = tune_neighborhood_size(func_kld, NodeKLDAutoMon, conf, 2 * conf["k"], data_generator, conf["sliding_window_size"], update_local_vector_concatenated_frequency_vectors)
        conf["neighborhood_size"] = tuned_neighborhood_size

        logging.info("\n ###################### Start KLD AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeKLDAutoMon, CoordinatorAutoMon, conf, 2 * conf["k"], func_kld)
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_concatenated_frequency_vectors)

        plot_figures(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    parent_test_folder = start_test("test_dimension_impact_kld_air_quality")
    end_test()  # To close the logging

    dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]

    for dimension in dimensions:
        test_dimension(dimension, parent_test_folder)
