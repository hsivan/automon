import os
os.environ['AUTO_DIFFERENTIATION_TOOL'] = 'AutoGrad'
from utils.nodes_automon import NodeRozenbrockAutoMon
from automon.automon.tune_neighborhood_size import tune_neighborhood_size
from utils.data_generator import DataGeneratorRozenbrock
from automon.coordinator_common import SlackType, SyncType
from automon.automon.coordinator_automon import CoordinatorAutoMon
from utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file, read_config_file
from utils.stats_analysis_utils import plot_monitoring_stats, plot_impact_of_neighborhood_size_on_violations
import logging
import numpy as np
import traceback
from concurrent.futures import ProcessPoolExecutor
from utils.object_factory import get_objects
from tests.visualization.plot_neighborhood_impact import plot_neighborhood_size_error_bound_connection_avg


def neighborhood_size_impact(experiment_folder, error_bound):
    neighborhood_sizes = np.arange(0.01, 0.255, 0.005)

    test_folder = experiment_folder + "/thresh_" + str(error_bound).replace(".", "_")
    test_folder = start_test("neighborhood_size_impact_rozenbrock", test_folder)
    conf = read_config_file(experiment_folder)
    conf['error_bound'] = error_bound
    write_config_to_file(test_folder, conf)

    # Create data generator from file
    data_generator = DataGeneratorRozenbrock(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                             data_file_name="data_file.txt", d=conf["d"], test_folder=experiment_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

    coordinator, nodes = get_objects(NodeRozenbrockAutoMon, CoordinatorAutoMon, conf)
    tune_neighborhood_size(coordinator, nodes, conf, data_generator)
    with open(test_folder + "/tuned_neighborhood_size.txt", "a") as f:
        f.write("tuned_neighborhood_size: " + str(coordinator.neighborhood_size))

    end_test()  # To close the logging

    for neighborhood_size in neighborhood_sizes:
        try:
            sub_test_folder = test_folder + "/domain_" + str(neighborhood_size).replace(".", "_")
            sub_test_folder = start_test("neighborhood_size_impact_rozenbrock", sub_test_folder)

            conf['neighborhood_size'] = neighborhood_size
            write_config_to_file(sub_test_folder, conf)

            logging.info("\n###################### Start Rozenbrock AutoMon test ######################")
            data_generator.reset()
            coordinator, nodes = get_objects(NodeRozenbrockAutoMon, CoordinatorAutoMon, conf)
            coordinator.b_fix_neighborhood_dynamically = False  # Should not change neighborhood size dynamically in this experiment
            run_test(data_generator, coordinator, nodes, sub_test_folder)

            plot_monitoring_stats(sub_test_folder)
            end_test()

        except Exception as e:
            logging.info(traceback.print_exc())
            end_test()
            raise e

    plot_impact_of_neighborhood_size_on_violations(test_folder)


if __name__ == "__main__":
    error_bounds = np.arange(0.05, 1.5, 0.1)
    num_experiments = 5

    parent_test_folder = start_test("optimal_and_tuned_neighborhood_rozenbrock")
    end_test()  # To close the logging

    for experiment_idx in range(num_experiments):
        experiment_folder = parent_test_folder + "/" + str(experiment_idx)
        test_folder = start_test("experiment_" + str(experiment_idx), experiment_folder)
        end_test()  # To close the logging

        # Generate basic config (each test should change the error bound and neighborhood size accordingly), and save it to file
        conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, d=2,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          neighborhood_size=1.0, num_iterations_for_tuning=200)
        write_config_to_file(experiment_folder, conf)
        data_generator = DataGeneratorRozenbrock(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                 d=conf["d"], test_folder=experiment_folder, num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

        executor = ProcessPoolExecutor(max_workers=20)

        for idx, error_bound in enumerate(error_bounds):
            executor.submit(neighborhood_size_impact, experiment_folder, error_bound)
        executor.shutdown()

    plot_neighborhood_size_error_bound_connection_avg(parent_test_folder, "Rozenbrock")
