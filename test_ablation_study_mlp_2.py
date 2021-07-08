from auto_mon_monitoring.node_mlp_auto_mon import NodeMlpAutoMon
from auto_mon_monitoring.tune_neighborhood_size import tune_neighborhood_size
from coordinators.coordinator_rlv import CoordinatorRLV
from data_generator import DataGeneratorMlp
from coordinators.coordinator_common import SlackType, SyncType
from coordinators.coordinator_auto_mon import CoordinatorAutoMon, DomainType
from functions_to_update_local_vector import update_local_vector_average
from rlv_monitoring.node_mlp_rlv import NodeMlpRLV
from test_figures.plot_monitoring_stats_ablation_study import plot_monitoring_stats_graph_and_barchart, \
    plot_monitoring_stats_barchart
from test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from stats_analysis_utils import plot_figures
import logging
from jax_mlp import train_net, load_net, draw_f_approx_contour_and_node_trail
from object_factory import get_objects
from functions_to_monitor import func_mlp, set_net_params

if __name__ == "__main__":
    try:
        test_folder = start_test("ablation_study_mlp_2")

        # Generate data and save to file or read from file

        '''
        data_folder = 'datasets/MLP_2/'
        conf = read_config_file(data_folder)
        write_config_to_file(test_folder, conf)
        net_params, net_apply = load_net(data_folder)
        set_net_params(net_params, net_apply)
        data_generator = DataGenerator(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name="data_file.txt", test_folder=data_folder, num_iterations_for_tuning=200)
        '''

        conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, k=2, error_bound=0.15,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          domain_type=DomainType.Relative.value, neighborhood_size=0.5)
        write_config_to_file(test_folder, conf)
        net_params, net_apply = train_net(test_folder, num_train_iter=20000, step_size=1e-4)
        set_net_params(net_params, net_apply)
        data_generator = DataGeneratorMlp(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], k=conf["k"], test_folder=test_folder, num_iterations_for_tuning=200)

        tuned_neighborhood_size = tune_neighborhood_size(func_mlp, NodeMlpAutoMon, conf, conf["k"], data_generator, conf["sliding_window_size"], update_local_vector_average)
        conf["neighborhood_size"] = tuned_neighborhood_size

        draw_f_approx_contour_and_node_trail(net_apply, net_params, data_generator.data, test_folder)

        logging.info("\n ###################### Start MLP RLV test  (no ADCD no slack) ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeMlpRLV, CoordinatorRLV, conf, conf["k"], func_mlp)
        coordinator.coordinator_name = "no ADCD no slack"
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_average)

        logging.info("\n ###################### Start MLP RLV test  (no ADCD) ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeMlpRLV, CoordinatorRLV, conf, conf["k"], func_mlp)
        coordinator.coordinator_name = "no ADCD"
        coordinator.slack_type = SlackType.Drift
        coordinator.sync_type = SyncType.LazyLRU
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_average)

        logging.info("\n ###################### Start MLP AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeMlpAutoMon, CoordinatorAutoMon, conf, conf["k"], func_mlp)
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_average)

        plot_figures(test_folder)
        plot_monitoring_stats_graph_and_barchart(test_folder, "mlp_2", test_folder + "/")
        plot_monitoring_stats_barchart(test_folder, "mlp_2", test_folder + "/")

    finally:
        end_test()
