from automon.coordinator_common import SlackType, SyncType
from automon.rlv.coordinator_rlv import CoordinatorRLV
from automon.data_generator import DataGeneratorKldAirQuality, DataGeneratorQuadratic, DataGeneratorRozenbrock
from automon.rlv.nodes_rlv import NodeKLDRLV, NodeQuadraticRLV, NodeRozenbrockRLV
from automon.test_utils import start_test, end_test, get_config
from automon.functions_to_monitor import set_H
from tests.regression.regression_test_utils import compare_results
import numpy as np
from automon.object_factory import get_objects
from automon.test_utils import run_test

regression_test_files_folder = "./regression_test_files_rlv/"


def test_func(func_name, NodeClass, data_generator, conf, test_folder, sliding_window_size):
    print("\nRun " + func_name + " test with No Slack and Eager Sync")
    np.random.seed(seed=1)
    data_generator.reset()
    conf["slack_type"], conf["sync_type"] = SlackType.NoSlack.value, SyncType.Eager.value
    coordinator, nodes = get_objects(NodeClass, CoordinatorRLV, conf)
    sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder, sliding_window_size)
    compare_results(regression_test_files_folder, sync_history, msg_counters, func_name, "no", "eager")


if __name__ == "__main__":
    try:
        test_folder = start_test("rlv_regression")


        conf = get_config(num_nodes=10, num_iterations=200, sliding_window_size=100, d=20, error_bound=0.05,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, domain=(0, 1))
        # Read data from test file
        data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                    data_file_name=regression_test_files_folder + "data_file_kld.txt", d=conf["d"])
        test_func("kld", NodeKLDRLV, data_generator, conf, test_folder, conf["sliding_window_size"])


        conf = get_config(num_nodes=10, num_iterations=1000, sliding_window_size=5, d=10, error_bound=2,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        # Read data from test file
        data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                data_file_name=regression_test_files_folder + "data_file_quadratic.txt", d=conf["d"])
        H = np.loadtxt(regression_test_files_folder + 'H_matrix.txt', dtype=np.float32)
        set_H(conf["d"], H)
        test_func("quadratic", NodeQuadraticRLV, data_generator, conf, test_folder, conf["sliding_window_size"])


        conf = get_config(num_nodes=3, num_iterations=500, sliding_window_size=10, d=2, error_bound=10,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        # Read data from test file
        data_generator = DataGeneratorRozenbrock(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                 data_file_name=regression_test_files_folder + "data_file_rozenbrock.txt", d=conf["d"])
        test_func("rozenbrock", NodeRozenbrockRLV, data_generator, conf, test_folder, conf["sliding_window_size"])

    finally:
        end_test()