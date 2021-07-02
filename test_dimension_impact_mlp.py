from auto_mon_monitoring.node_mlp_auto_mon import NodeMlpAutoMon
from auto_mon_monitoring.tune_neighborhood_size import tune_neighborhood_size
from coordinators.coordinator_auto_mon import CoordinatorAutoMon, DomainType
from data_generator import DataGeneratorMlp
from coordinators.coordinator_common import SlackType, SyncType
from functions_to_update_local_vector import update_local_vector_average
from test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from stats_analysis_utils import plot_figures
import logging
from object_factory import get_objects
from jax_mlp import train_net
from functions_to_monitor import func_mlp, set_net_params


def test_dimension(dimension, parent_test_folder):

    try:
        test_folder = parent_test_folder + "/dimension_" + str(dimension)
        test_folder = start_test("dimension_" + str(dimension), test_folder)

        conf = get_config(num_nodes=12, num_iterations=1020, sliding_window_size=20, k=dimension, error_bound=0.2,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          domain_type=DomainType.Relative.value)
        write_config_to_file(test_folder, conf)

        net_params, net_apply = train_net(test_folder, dimension, 15000, 1e-4)
        set_net_params(net_params, net_apply)
        data_generator = DataGeneratorMlp(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                          k=conf["k"], test_folder=test_folder, num_iterations_for_tuning=200)

        tuned_neighborhood_size = tune_neighborhood_size(func_mlp, NodeMlpAutoMon, conf, conf["k"], data_generator, conf["sliding_window_size"], update_local_vector_average)
        conf["neighborhood_size"] = tuned_neighborhood_size

        logging.info("\n ###################### Start DNN exp AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeMlpAutoMon, CoordinatorAutoMon, conf, conf["k"], func_mlp)
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_average)

        plot_figures(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    parent_test_folder = start_test("test_dimension_impact_mlp")
    end_test()  # To close the logging

    dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]

    for dimension in dimensions:
        test_dimension(dimension, parent_test_folder)
