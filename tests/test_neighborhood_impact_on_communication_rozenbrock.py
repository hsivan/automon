import os
os.environ['AUTO_DIFFERENTIATION_TOOL'] = 'AutoGrad'
from automon.automon.node_common_automon import NodeCommonAutoMon
from automon_utils.functions_to_monitor import func_rozenbrock
from automon_utils.data_generator import DataGeneratorRozenbrock
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon_utils.test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from automon_utils.stats_analysis_utils import plot_monitoring_stats, get_neighborhood_size_error_bound_connection
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from automon_utils.object_factory import get_objects
from tests.visualization.plot_neighborhood_impact import plot_communication_or_violation_error_bound_connection


def neighborhood_size_impact(experiment_folder, test_folder, neighborhood_sizes, prefixes):
    conf = read_config_file(test_folder)
    data_generator = DataGeneratorRozenbrock(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                             data_file_name="data_file.txt", d=conf["d"], test_folder=experiment_folder,
                                             num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])

    for i in range(len(neighborhood_sizes)):
        try:
            sub_test_folder = test_folder + "/" + prefixes[i] + "_neighborhood_" + str(neighborhood_sizes[i]).replace(".", "_")
            sub_test_folder = start_test("neighborhood_size_impact_rozenbrock", sub_test_folder)

            conf['neighborhood_size'] = neighborhood_sizes[i]
            write_config_to_file(sub_test_folder, conf)

            logging.info("\n###################### Start Rozenbrock AutoMon test ######################")
            data_generator.reset()
            coordinator, nodes = get_objects(NodeCommonAutoMon, CoordinatorAutoMon, conf, func_rozenbrock)
            if prefixes[i] != "tuned":
                coordinator.b_fix_neighborhood_dynamically = False  # Should not change neighborhood size dynamically for optimal and fixed neighborhood sizes
            run_test(data_generator, coordinator, nodes, sub_test_folder)

            plot_monitoring_stats(sub_test_folder)
            end_test()

        except Exception as e:
            logging.info(traceback.print_exc())
            end_test()
            raise e


def get_optimal_neighborhood_sizes_from_full_test(experiment_folders):
    optimal_neighborhood_sizes_experiments = []
    tuned_neighborhood_sizes_experiments = []
    for experiment in experiment_folders:
        error_bounds, optimal_neighborhood_sizes, tuned_neighborhood_sizes = get_neighborhood_size_error_bound_connection(experiment)
        optimal_neighborhood_sizes_experiments.append(optimal_neighborhood_sizes)
        tuned_neighborhood_sizes_experiments.append(tuned_neighborhood_sizes)

    return error_bounds, optimal_neighborhood_sizes_experiments, tuned_neighborhood_sizes_experiments


if __name__ == "__main__":

    # Use the output folder of test_optimal_and_tuned_neighborhood_rozenbrock.py to get the optimal and tuned neighborhood size per error bound
    full_tuning_test_folder = "./test_results/results_optimal_and_tuned_neighborhood_rozenbrock_2021-04-05_08-48-07/"
    full_tuning_test_experiment_folders = [full_tuning_test_folder + sub_folder for sub_folder in os.listdir(full_tuning_test_folder) if os.path.isdir(full_tuning_test_folder + sub_folder)]
    num_experiments = len(full_tuning_test_experiment_folders)
    error_bounds, optimal_neighborhood_size_arr, tuned_neighborhood_size_arr = get_optimal_neighborhood_sizes_from_full_test(full_tuning_test_experiment_folders)

    parent_test_folder = start_test("comm_neighborhood_rozen")
    end_test()  # To close the logging

    for experiment_idx in range(num_experiments):
        experiment_folder = parent_test_folder + "/" + str(experiment_idx)
        experiment_folder = start_test("experiment_" + str(experiment_idx), experiment_folder)
        end_test()  # To close the logging

        # Generate basic config (each test should change the error bound and neighborhood size accordingly), and save it to file
        conf = read_config_file(full_tuning_test_experiment_folders[experiment_idx])
        write_config_to_file(experiment_folder, conf)
        # Generate data and save to file
        data_generator = DataGeneratorRozenbrock(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                 data_file_name="data_file.txt", d=conf["d"], test_folder=full_tuning_test_experiment_folders[experiment_idx],
                                                 num_iterations_for_tuning=conf["num_iterations_for_tuning"], sliding_window_size=conf["sliding_window_size"])
        data_generator.data_file_name = experiment_folder + "/data_file.txt"
        data_generator._save_data_to_file()

        executor = ProcessPoolExecutor(max_workers=20)

        for idx, error_bound in enumerate(error_bounds):

            test_folder = experiment_folder + "/thresh_" + str(error_bound).replace(".", "_")
            test_folder = start_test("neighborhood_size_impact_rozenbrock", test_folder)
            conf = read_config_file(experiment_folder)
            conf['error_bound'] = error_bound
            write_config_to_file(test_folder, conf)
            end_test()  # To close the logging

            optimal_neighborhood_size = optimal_neighborhood_size_arr[experiment_idx][idx]
            tuned_neighborhood_size = tuned_neighborhood_size_arr[experiment_idx][idx]
            fixed_neighborhood_sizes = [0.05, 0.5, 2.5]
            neighborhood_sizes = [optimal_neighborhood_size, tuned_neighborhood_size] + fixed_neighborhood_sizes
            prefixes = ["optimal", "tuned"] + [str(i) + "_fixed" for i in range(len(fixed_neighborhood_sizes))]

            executor.submit(neighborhood_size_impact, experiment_folder, test_folder, neighborhood_sizes, prefixes)

        executor.shutdown()

    plot_communication_or_violation_error_bound_connection(parent_test_folder, "Rozenbrock")
