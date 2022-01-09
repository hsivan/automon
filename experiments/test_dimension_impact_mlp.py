from automon.automon.automon_node import AutomonNode
from test_utils.tune_neighborhood_size import tune_neighborhood_size
from automon.automon.automon_coordinator import AutomonCoordinator
from test_utils.data_generator import DataGeneratorMlp
from automon.common_coordinator import SlackType, SyncType
from test_utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging
from test_utils.object_factory import get_objects
from test_utils.jax_mlp import train_net
from test_utils.functions_to_monitor import set_net_params, func_mlp


def test_dimension(dimension, parent_test_folder):

    try:
        test_folder = parent_test_folder + "/dimension_" + str(dimension)
        test_folder = start_test("dimension_" + str(dimension), test_folder)

        conf = get_config(num_nodes=12, num_iterations=1020, sliding_window_size=20, d=dimension, error_bound=0.2,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          neighborhood_size=1.0, num_iterations_for_tuning=200)
        write_config_to_file(test_folder, conf)

        net_params, net_apply = train_net(test_folder, dimension, 15000, 1e-4)
        set_net_params(net_params, net_apply)
        data_generator = DataGeneratorMlp(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                          d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

        logging.info("\n###################### Start DNN exp AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(AutomonNode, AutomonCoordinator, conf, func_mlp)
        tune_neighborhood_size(coordinator, nodes, conf, data_generator)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    parent_test_folder = start_test("test_dimension_impact_mlp")
    end_test()  # To close the logging

    dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]

    for dimension in dimensions:
        test_dimension(dimension, parent_test_folder)
