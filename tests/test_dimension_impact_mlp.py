from utils.nodes_automon import NodeMlpAutoMon
from automon.automon.tune_neighborhood_size import tune_neighborhood_size
from automon.automon.coordinator_automon import CoordinatorAutoMon
from utils.data_generator import DataGeneratorMlp
from automon.coordinator_common import SlackType, SyncType
from utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from utils.stats_analysis_utils import plot_monitoring_stats
import logging
from utils.object_factory import get_objects
from utils.jax_mlp import train_net
from utils.functions_to_monitor import set_net_params


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
        coordinator, nodes = get_objects(NodeMlpAutoMon, CoordinatorAutoMon, conf)
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
