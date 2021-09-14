from automon.automon.nodes_automon import NodeMlpAutoMon
from automon.automon.tune_neighborhood_size import tune_neighborhood_size
from automon.data_generator import DataGeneratorMlp
from automon.automon.coordinator_automon import CoordinatorAutoMon
from tests.visualization.plot_neighborhood_impact import plot_neighborhood_size_error_bound_connection_avg
from automon.test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from automon.stats_analysis_utils import plot_monitoring_stats, plot_impact_of_neighborhood_size_on_violations
import logging
import numpy as np
import traceback
from automon.object_factory import get_objects
from automon.functions_to_monitor import set_net_params
from automon.jax_mlp import load_net


def neighborhood_size_impact(experiment_folder, error_bound):
    neighborhood_sizes = np.arange(0.1, 1.0, 0.05)

    data_folder = '../datasets/MLP_2/'
    net_params, net_apply = load_net(data_folder)
    set_net_params(net_params, net_apply)

    test_folder = experiment_folder + "/thresh_" + str(error_bound).replace(".", "_")
    test_folder = start_test("neighborhood_size_impact_mlp_2", test_folder)
    conf = read_config_file(experiment_folder)
    conf['error_bound'] = error_bound
    write_config_to_file(test_folder, conf)

    # Create data generator from file
    data_generator = DataGeneratorMlp(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                      data_file_name="data_file.txt", test_folder=experiment_folder, d=conf["d"], num_iterations_for_tuning=conf["num_iterations_for_tuning"])

    coordinator, nodes = get_objects(NodeMlpAutoMon, CoordinatorAutoMon, conf)
    tune_neighborhood_size(coordinator, nodes, conf, data_generator)
    with open(test_folder + "/tuned_neighborhood_size.txt", "a") as f:
        f.write("tuned_neighborhood_size: " + str(coordinator.neighborhood_size))

    end_test()  # To close the logging

    for neighborhood_size in neighborhood_sizes:
        try:
            sub_test_folder = test_folder + "/domain_" + str(neighborhood_size).replace(".", "_")
            sub_test_folder = start_test("neighborhood_size_impact_mlp_2", sub_test_folder)

            conf['neighborhood_size'] = neighborhood_size
            write_config_to_file(sub_test_folder, conf)

            logging.info("\n###################### Start MLP AutoMon test ######################")
            data_generator.reset()
            coordinator, nodes = get_objects(NodeMlpAutoMon, CoordinatorAutoMon, conf)
            coordinator.b_fix_neighborhood_dynamically = False  # Should not change neighborhood size dynamically in this experiment
            run_test(data_generator, coordinator, nodes, sub_test_folder, conf["sliding_window_size"])

            plot_monitoring_stats(sub_test_folder)
            end_test()

        except Exception as e:
            logging.info(traceback.print_exc())
            end_test()
            raise e

    plot_impact_of_neighborhood_size_on_violations(test_folder)


if __name__ == "__main__":
    error_bounds = np.arange(0.005, 0.35, 0.015)
    num_experiments = 5

    parent_test_folder = start_test("optimal_and_tuned_neighborhood_mlp_2")
    end_test()  # To close the logging

    for experiment_idx in range(num_experiments):
        experiment_folder = parent_test_folder + "/" + str(experiment_idx)
        test_folder = start_test("experiment_" + str(experiment_idx), experiment_folder)
        end_test()  # To close the logging

        # Generate basic config (each test should change the error bound and neighborhood size accordingly), and save it to file
        data_folder = '../datasets/MLP_2/'
        conf = read_config_file(data_folder)
        write_config_to_file(experiment_folder, conf)
        # Generate data and save to file
        data_generator = DataGeneratorMlp(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                          d=conf["d"], test_folder=experiment_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"])

        for idx, error_bound in enumerate(error_bounds):
            neighborhood_size_impact(experiment_folder, error_bound)

    plot_neighborhood_size_error_bound_connection_avg(parent_test_folder, "MLP-2")
