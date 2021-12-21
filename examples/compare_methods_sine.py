from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.automon.node_common_automon import NodeCommonAutoMon
from utils.data_generator import DataGeneratorSine
from automon.coordinator_common import SlackType, SyncType
from utils.functions_to_monitor import func_sine
from utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from utils.stats_analysis_utils import plot_monitoring_stats
import logging
from utils.object_factory import get_objects
import numpy as np

if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_sine")
        np.random.seed(0)

        conf = get_config(num_nodes=10, num_iterations=4000, sliding_window_size=20, d=1, error_bound=1.5,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          neighborhood_size=3)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorSine(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                           data_file_name="../tests/visualization/sine_monitoring/data_file.txt", d=conf["d"], test_folder="./", sliding_window_size=conf["sliding_window_size"])

        logging.info("\n###################### Start Sine AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeCommonAutoMon, CoordinatorAutoMon, conf, func_sine)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)

    finally:
        end_test()
