from automon import AutomonNode, AutomonCoordinator, RlvNode, RlvCoordinator, SlackType, SyncType
from test_utils.functions_to_monitor import get_func_quadratic
from test_utils.data_generator import DataGeneratorQuadratic
from test_utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging
import numpy as np
from test_utils.object_factory import get_objects

if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_quadratic")

        # Generate data and save to file or read from file. Domain is None (all R) for both AutoMon and RLV.

        conf = get_config(num_nodes=4, num_iterations=1010, sliding_window_size=10, d=4, error_bound=0.05,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        write_config_to_file(test_folder, conf)
        data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder, sliding_window_size=conf["sliding_window_size"])

        H = np.random.randn(conf["d"], conf["d"]).astype(np.float32)
        np.savetxt(test_folder + '/H_matrix.txt', H, fmt='%f')
        func_quadratic = get_func_quadratic(H)

        logging.info("\n###################### Start KLD RLV test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(RlvNode, RlvCoordinator, conf, func_quadratic)
        run_test(data_generator, coordinator, nodes, test_folder)

        logging.info("\n###################### Start quadratic AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(AutomonNode, AutomonCoordinator, conf, func_quadratic)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
