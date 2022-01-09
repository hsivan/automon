from automon import AutomonNode, AutomonCoordinator
from test_utils.tune_neighborhood_size import tune_neighborhood_size
from test_utils.data_generator import DataGeneratorMlp
from test_utils.test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging
from test_utils.jax_mlp import load_net
from test_utils.object_factory import get_objects
from test_utils.functions_to_monitor import set_net_params, func_mlp
from experiments.visualization.plot_num_nodes_impact import plot_num_nodes_impact_on_communication


def test_num_nodes(num_nodes, parent_test_folder):
    conf = read_config_file(parent_test_folder)

    try:
        test_folder = parent_test_folder + "/num_nodes_" + str(num_nodes)
        test_folder = start_test("num_nodes_" + str(num_nodes), test_folder)

        conf["num_nodes"] = num_nodes
        write_config_to_file(test_folder, conf)

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
    parent_test_folder = start_test("test_num_nodes_impact_mlp_40")
    end_test()  # To close the logging

    data_folder = '../datasets/MLP_40/'
    conf = read_config_file(data_folder)
    write_config_to_file(parent_test_folder, conf)
    net_params, net_apply = load_net(data_folder)
    set_net_params(net_params, net_apply)

    num_nodes_arr = [10, 20, 40, 60, 100, 500]

    for num_nodes in num_nodes_arr:
        test_num_nodes(num_nodes, parent_test_folder)

    plot_num_nodes_impact_on_communication(parent_test_folder)
