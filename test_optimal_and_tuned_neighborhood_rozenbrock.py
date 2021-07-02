import os
os.environ['AUTO_GRAD_TOOL'] = 'AutoGrad'
from auto_mon_monitoring.node_rosenbrock_auto_mon import NodeRozenbrockAutoMon
from auto_mon_monitoring.tune_neighborhood_size import tune_neighborhood_size
from data_generator import DataGenerator, DataGeneratorRozenbrock
from coordinators.coordinator_common import SlackType, SyncType
from coordinators.coordinator_auto_mon import CoordinatorAutoMon, DomainType
from test_utils import start_test, end_test, run_test, get_config, write_config_to_file, read_config_file
from stats_analysis_utils import plot_figures, plot_impact_of_neighborhood_size_on_violations, plot_neighborhood_size_error_bound_connection_avg
import logging
import numpy as np
import traceback
from concurrent.futures import ProcessPoolExecutor
from object_factory import get_objects
from functions_to_monitor import func_rozenbrock
from functions_to_update_local_vector import update_local_vector_average


def neighborhood_size_impact(experiment_folder, error_bound):
    neighborhood_sizes = np.arange(0.01, 0.255, 0.005)

    test_folder = experiment_folder + "/thresh_" + str(error_bound).replace(".", "_")
    test_folder = start_test("neighborhood_size_impact_rozenbrock", test_folder)
    conf = read_config_file(experiment_folder)
    conf['error_bound'] = error_bound
    write_config_to_file(test_folder, conf)

    # Create data generator from file
    data_generator = DataGenerator(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                   data_file_name="data_file.txt", test_folder=experiment_folder, num_iterations_for_tuning=200)

    tuned_neighborhood_size = tune_neighborhood_size(func_rozenbrock, NodeRozenbrockAutoMon, conf, 2 * conf["k"], data_generator, conf["sliding_window_size"], update_local_vector_average)
    with open(test_folder + "/tuned_neighborhood_size.txt", "a") as f:
        f.write("tuned_neighborhood_size: " + str(tuned_neighborhood_size))

    end_test()  # To close the logging

    for neighborhood_size in neighborhood_sizes:
        try:
            sub_test_folder = test_folder + "/domain_" + str(neighborhood_size).replace(".", "_")
            sub_test_folder = start_test("neighborhood_size_impact_rozenbrock", sub_test_folder)

            conf['neighborhood_size'] = neighborhood_size
            write_config_to_file(sub_test_folder, conf)

            logging.info("\n ###################### Start Rozenbrock AutoMon test ######################")
            data_generator.reset()
            coordinator, nodes, verifier = get_objects(NodeRozenbrockAutoMon, CoordinatorAutoMon, conf, 2 * conf["k"], func_rozenbrock)
            run_test(data_generator, coordinator, nodes, verifier, sub_test_folder, conf["sliding_window_size"], update_local_vector_average)

            plot_figures(sub_test_folder)
            end_test()

        except Exception as e:
            logging.info(traceback.print_exc())
            end_test()

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
        conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, k=1,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          domain_type=DomainType.Relative.value)
        write_config_to_file(experiment_folder, conf)
        data_generator = DataGeneratorRozenbrock(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                 k=conf["k"], test_folder=experiment_folder, num_iterations_for_tuning=200)

        executor = ProcessPoolExecutor(max_workers=20)
        futures_arr = []

        for idx, error_bound in enumerate(error_bounds):
            future = executor.submit(neighborhood_size_impact, experiment_folder, error_bound)
            futures_arr.append(future)
        executor.shutdown()

    plot_neighborhood_size_error_bound_connection_avg(parent_test_folder)

