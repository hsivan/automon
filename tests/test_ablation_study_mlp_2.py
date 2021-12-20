from utils.nodes_automon import NodeMlpAutoMon
from utils.tune_neighborhood_size import tune_neighborhood_size
from automon.rlv.coordinator_rlv import CoordinatorRLV
from utils.data_generator import DataGeneratorMlp
from automon.coordinator_common import SlackType, SyncType
from automon.automon.coordinator_automon import CoordinatorAutoMon
from utils.nodes_rlv import NodeMlpRLV
from tests.visualization.plot_monitoring_stats_ablation_study import plot_monitoring_stats_graph_and_barchart, plot_monitoring_stats_barchart
from utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from utils.stats_analysis_utils import plot_monitoring_stats
import logging
from utils.jax_mlp import train_net, draw_f_approx_contour_and_node_trail
from utils.object_factory import get_objects
from utils.functions_to_monitor import set_net_params

if __name__ == "__main__":
    try:
        test_folder = start_test("ablation_study_mlp_2")

        # Generate data and save to file or read from file

        '''
        data_folder = '../datasets/MLP_2/'
        conf = read_config_file(data_folder)
        write_config_to_file(test_folder, conf)
        net_params, net_apply = load_net(data_folder)
        set_net_params(net_params, net_apply)
        data_generator = DataGeneratorMlp(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name="data_file.txt", test_folder=data_folder, d=conf["d"], num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])
        '''

        conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, d=2, error_bound=0.15,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          neighborhood_size=0.5, num_iterations_for_tuning=200)
        write_config_to_file(test_folder, conf)
        net_params, net_apply = train_net(test_folder, num_train_iter=20000, step_size=1e-4)
        set_net_params(net_params, net_apply)
        data_generator = DataGeneratorMlp(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

        draw_f_approx_contour_and_node_trail(net_apply, net_params, data_generator.data, test_folder)

        logging.info("\n###################### Start MLP RLV test  (no ADCD no slack) ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeMlpRLV, CoordinatorRLV, conf)
        coordinator.coordinator_name = "no ADCD no slack"
        run_test(data_generator, coordinator, nodes, test_folder)

        logging.info("\n###################### Start MLP RLV test  (no ADCD) ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeMlpRLV, CoordinatorRLV, conf)
        coordinator.coordinator_name = "no ADCD"
        coordinator.slack_type = SlackType.Drift
        coordinator.sync_type = SyncType.LazyLRU
        run_test(data_generator, coordinator, nodes, test_folder)

        logging.info("\n###################### Start MLP AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeMlpAutoMon, CoordinatorAutoMon, conf)
        tune_neighborhood_size(coordinator, nodes, conf, data_generator)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)
        plot_monitoring_stats_graph_and_barchart(test_folder, "mlp_2", test_folder + "/")
        plot_monitoring_stats_barchart(test_folder, "mlp_2", test_folder + "/")

    finally:
        end_test()
