from auto_mon_monitoring.node_quadratic_auto_mon import NodeQuadraticAutoMon
from coordinators.coordinator_auto_mon import CoordinatorAutoMon
from data_generator import DataGenerator, DataGeneratorQuadratic
from coordinators.coordinator_common import SlackType, SyncType
from functions_to_update_local_vector import update_local_vector_average
from test_utils import start_test, end_test, run_test, get_config, write_config_to_file, read_config_file
from stats_analysis_utils import plot_figures
import logging
from object_factory import get_objects
from functions_to_monitor import func_quadratic, set_H, get_H
import numpy as np
from test_figures.plot_error_communication_tradeoff import plot_max_error_vs_communication


def test_error_bounds(error_bound, parent_test_folder):
    conf = read_config_file(parent_test_folder)
    H = np.loadtxt(parent_test_folder + '/H_matrix.txt', dtype=np.float32)
    set_H(conf["k"], H)
    data_generator = DataGenerator(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                   data_file_name="data_file.txt", test_folder=parent_test_folder)

    try:
        test_folder = parent_test_folder + "/threshold_" + str(error_bound)
        test_folder = start_test("error_bound_" + str(error_bound), test_folder)

        conf["error_bound"] = error_bound
        write_config_to_file(test_folder, conf)

        logging.info("\n ###################### Start AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeQuadraticAutoMon, CoordinatorAutoMon, conf, conf["k"], func_quadratic)
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_average)

        plot_figures(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    parent_test_folder = start_test("test_max_error_vs_communication_quadratic")
    end_test()  # To close the logging

    conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, k=40,
                      slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
    write_config_to_file(parent_test_folder, conf)
    set_H(conf["k"])
    H = get_H().copy()
    np.savetxt(parent_test_folder + '/H_matrix.txt', H, fmt='%f')

    data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], k=conf["k"], test_folder=parent_test_folder)

    error_bounds = [0.015, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 1.0, 1.5]

    for error_bound in error_bounds:
        test_error_bounds(error_bound, parent_test_folder)

    plot_max_error_vs_communication(parent_test_folder, "Quadratic")
