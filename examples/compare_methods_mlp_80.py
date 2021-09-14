from automon.automon.nodes_automon import NodeMlpAutoMon
from automon.automon.tune_neighborhood_size import tune_neighborhood_size
from automon.rlv.coordinator_rlv import CoordinatorRLV
from automon.data_generator import DataGeneratorMlp
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.rlv.nodes_rlv import NodeMlpRLV
from automon.test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from automon.stats_analysis_utils import plot_monitoring_stats
import logging
from automon.jax_mlp import load_net
from automon.object_factory import get_objects
from automon.functions_to_monitor import set_net_params

if __name__ == "__main__":
    try:
        x_dim = 80
        test_folder = start_test("compare_methods_mlp_" + str(x_dim))

        # Generate data and save to file or read from file

        data_folder = '../datasets/MLP_80/'
        conf = read_config_file(data_folder)
        write_config_to_file(test_folder, conf)
        net_params, net_apply = load_net(data_folder)
        set_net_params(net_params, net_apply)
        data_generator = DataGeneratorMlp(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name="data_file.txt",
                                          d=conf["d"], test_folder=data_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"])

        '''
        conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, d=x_dim, error_bound=0.2,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          domain_type=DomainType.Relative.value, neighborhood_size=0.4, num_iterations_for_tuning=200)
        write_config_to_file(test_folder, conf)
        net_params, net_apply = train_net(test_folder, x_dim, 15000, 1e-4)
        set_net_params(net_params, net_apply)
        data_generator = DataGeneratorMlp(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                          d=conf["d"], test_folder=test_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"])
        '''

        logging.info("\n###################### Start MLP RLV test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeMlpRLV, CoordinatorRLV, conf)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        logging.info("\n###################### Start MLP AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeMlpAutoMon, CoordinatorAutoMon, conf)
        tune_neighborhood_size(coordinator, nodes, conf, data_generator)
        run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
