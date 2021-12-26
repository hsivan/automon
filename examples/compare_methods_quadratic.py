from automon.automon.node_common_automon import NodeCommonAutoMon
from automon.rlv.node_common_rlv import NodeCommonRLV
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.rlv.coordinator_rlv import CoordinatorRLV
from test_utils.functions_to_monitor import set_H, get_H, func_quadratic
from test_utils.data_generator import DataGeneratorQuadratic
from automon.coordinator_common import SlackType, SyncType
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
        set_H(conf["d"])
        H = get_H().copy()
        data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder, sliding_window_size=conf["sliding_window_size"])

        np.savetxt(test_folder + '/H_matrix.txt', H, fmt='%f')

        logging.info("\n###################### Start KLD RLV test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeCommonRLV, CoordinatorRLV, conf, func_quadratic)
        run_test(data_generator, coordinator, nodes, test_folder)

        logging.info("\n###################### Start quadratic AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeCommonAutoMon, CoordinatorAutoMon, conf, func_quadratic)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
