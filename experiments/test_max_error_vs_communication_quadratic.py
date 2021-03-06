from automon import AutomonNode, AutomonCoordinator
from test_utils.data_generator import DataGeneratorQuadratic
from test_utils.test_utils import start_test, end_test, run_test, write_config_to_file, read_config_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging
from test_utils.object_factory import get_objects
from test_utils.functions_to_monitor import get_func_quadratic
import numpy as np
from experiments.visualization.plot_error_communication_tradeoff import plot_max_error_vs_communication


def test_error_bounds(error_bound, parent_test_folder):
    conf = read_config_file(parent_test_folder)
    H = np.loadtxt(parent_test_folder + '/H_matrix.txt', dtype=np.float32)
    func_quadratic = get_func_quadratic(H)
    data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name="data_file.txt", test_folder=parent_test_folder, d=conf["d"], sliding_window_size=conf["sliding_window_size"])

    try:
        test_folder = parent_test_folder + "/threshold_" + str(error_bound)
        test_folder = start_test("error_bound_" + str(error_bound), test_folder)

        conf["error_bound"] = error_bound
        write_config_to_file(test_folder, conf)

        logging.info("\n###################### Start AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(AutomonNode, AutomonCoordinator, conf, func_quadratic)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()


if __name__ == "__main__":

    parent_test_folder = start_test("test_max_error_vs_communication_quadratic")
    end_test()  # To close the logging

    '''conf = get_config(num_nodes=10, num_iterations=1020, sliding_window_size=20, d=40, slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
    write_config_to_file(parent_test_folder, conf)
    H = np.random.randn(conf["d"], conf["d"]).astype(np.float32)
    np.savetxt(parent_test_folder + '/H_matrix.txt', H, fmt='%f')
    data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=parent_test_folder, sliding_window_size=conf["sliding_window_size"])'''

    data_folder = '../datasets/quadratic/'
    conf = read_config_file(data_folder)
    H = np.loadtxt(data_folder + 'H_matrix.txt', dtype=np.float32)
    np.savetxt(parent_test_folder + '/H_matrix.txt', H, fmt='%f')
    write_config_to_file(parent_test_folder, conf)
    data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], data_file_name=data_folder + "data_file.txt", sliding_window_size=conf["sliding_window_size"])
    data_generator.data_file_name = parent_test_folder + "/data_file.txt"
    data_generator._save_data_to_file()

    error_bounds = [0.015, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 1.0]

    for error_bound in error_bounds:
        test_error_bounds(error_bound, parent_test_folder)

    plot_max_error_vs_communication(parent_test_folder, "Quadratic")
