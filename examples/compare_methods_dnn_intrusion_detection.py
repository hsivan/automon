from automon import AutomonNode, AutomonCoordinator, RlvNode, RlvCoordinator
from test_utils.functions_to_monitor import set_net_params, func_dnn_intrusion_detection
from test_utils.data_generator import DataGeneratorDnnIntrusionDetection
from test_utils.test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging
from test_utils.jax_dnn_intrusion_detection import load_net
from test_utils.object_factory import get_objects

if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_dnn_intrusion_detection")

        # Generate data and save to file or read from file

        data_folder = '../datasets/intrusion_detection/'
        conf = read_config_file(data_folder)
        write_config_to_file(test_folder, conf)
        net_params, net_apply = load_net(data_folder)
        set_net_params(net_params, net_apply)
        data_generator = DataGeneratorDnnIntrusionDetection(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

        '''
        conf = get_config(num_nodes=9, num_iterations=302973, sliding_window_size=20, d=41, error_bound=0.005,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          neighborhood_size=0.5, num_iterations_for_tuning=500)
        write_config_to_file(test_folder, conf)
        net_params, net_apply = train_net(test_folder)
        set_net_params(net_params, net_apply)
        data_generator = DataGeneratorDnnIntrusionDetection(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])
        '''

        logging.info("\n###################### Start DNN intrusion_detection RLV test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(RlvNode, RlvCoordinator, conf, func_dnn_intrusion_detection)
        run_test(data_generator, coordinator, nodes, test_folder, b_single_sample_per_round=True)

        logging.info("\n###################### Start DNN intrusion_detection AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(AutomonNode, AutomonCoordinator, conf, func_dnn_intrusion_detection)
        run_test(data_generator, coordinator, nodes, test_folder, b_single_sample_per_round=True)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
