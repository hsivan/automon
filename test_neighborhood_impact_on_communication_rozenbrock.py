import os
os.environ['AUTO_GRAD_TOOL'] = 'AutoGrad'
from functions_to_update_local_vector import update_local_vector_average
from auto_mon_monitoring.node_rosenbrock_auto_mon import NodeRozenbrockAutoMon
from data_generator import DataGenerator
from coordinators.coordinator_auto_mon import CoordinatorAutoMon
from test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from stats_analysis_utils import plot_figures
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from object_factory import get_objects
from functions_to_monitor import func_rozenbrock
from test_neighborhood_impact_on_communication_mlp_2 import get_optimal_neighborhood_sizes_from_full_test


def neighborhood_size_impact(experiment_folder, test_folder, neighborhood_sizes, prefixes):
    conf = read_config_file(test_folder)
    data_generator = DataGenerator(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                   data_file_name="data_file.txt", test_folder=experiment_folder,
                                   num_iterations_for_tuning=200)

    for i in range(len(neighborhood_sizes)):
        try:
            sub_test_folder = test_folder + "/" + prefixes[i] + "_neighborhood_" + str(neighborhood_sizes[i]).replace(".", "_")
            sub_test_folder = start_test("neighborhood_size_impact_rozenbrock", sub_test_folder)

            conf['neighborhood_size'] = neighborhood_sizes[i]
            write_config_to_file(sub_test_folder, conf)

            logging.info("\n ###################### Start Rozenbrock AutoMon test ######################")
            data_generator.reset()
            coordinator, nodes, verifier = get_objects(NodeRozenbrockAutoMon, CoordinatorAutoMon, conf, 2 * conf["k"], func_rozenbrock)
            if prefixes[i] != "tuned":
                coordinator.b_fix_neighborhood_dynamically = False  # Should not change neighborhood size dynamically for optimal and fixed neighborhood sizes
            run_test(data_generator, coordinator, nodes, verifier, sub_test_folder, conf["sliding_window_size"], update_local_vector_average)

            plot_figures(sub_test_folder)
            end_test()

        except Exception as e:
            logging.info(traceback.print_exc())
            end_test()


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
        data_generator = DataGenerator(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                       data_file_name="data_file.txt", test_folder=full_tuning_test_experiment_folders[experiment_idx],
                                       num_iterations_for_tuning=200)
        data_generator.data_file_name = experiment_folder + "/data_file.txt"
        data_generator._save_data_to_file()

        executor = ProcessPoolExecutor(max_workers=20)
        futures_arr = []

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

            future = executor.submit(neighborhood_size_impact, experiment_folder, test_folder, neighborhood_sizes, prefixes)
            futures_arr.append(future)

        executor.shutdown()

