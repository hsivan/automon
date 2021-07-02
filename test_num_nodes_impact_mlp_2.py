from auto_mon_monitoring.node_mlp_auto_mon import NodeMlpAutoMon
from auto_mon_monitoring.tune_neighborhood_size import tune_neighborhood_size
from data_generator import DataGeneratorMlp
from coordinators.coordinator_auto_mon import CoordinatorAutoMon
from functions_to_update_local_vector import update_local_vector_average
from test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from stats_analysis_utils import plot_figures
import logging
from jax_mlp import load_net
from object_factory import get_objects
from functions_to_monitor import func_mlp, set_net_params
from test_figures.plot_num_nodes_impact import plot_num_nodes_impact_on_communication


def test_num_nodes(num_nodes, parent_test_folder):
    conf = read_config_file(parent_test_folder)

    try:
        test_folder = parent_test_folder + "/num_nodes_" + str(num_nodes)
        test_folder = start_test("num_nodes_" + str(num_nodes), test_folder)

        conf["num_nodes"] = num_nodes
        write_config_to_file(test_folder, conf)

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
    parent_test_folder = start_test("test_num_nodes_impact_mlp_2")
    end_test()  # To close the logging

    data_folder = 'datasets/MLP_2/'
    conf = read_config_file(data_folder)
    write_config_to_file(parent_test_folder, conf)
    net_params, net_apply = load_net(data_folder)
    set_net_params(net_params, net_apply)

    num_nodes_arr = [10, 20, 40, 60, 100]

    for num_nodes in num_nodes_arr:
        test_num_nodes(num_nodes, parent_test_folder)

    plot_num_nodes_impact_on_communication(parent_test_folder)
