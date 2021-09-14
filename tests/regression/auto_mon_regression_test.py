from automon.automon.nodes_automon import NodeEntropyAutoMon, NodeInnerProductAutoMon, NodeKLDAutoMon, \
    NodeQuadraticAutoMon, NodeRozenbrockAutoMon, NodeVarianceAutoMon
from automon.automon.coordinator_automon import CoordinatorAutoMon, DomainType
from automon.coordinator_common import SlackType, SyncType
from automon.data_generator import DataGeneratorEntropy, DataGeneratorVariance, DataGeneratorInnerProduct, \
    DataGeneratorKldAirQuality, DataGeneratorQuadratic, DataGeneratorRozenbrock
from automon.test_utils import start_test, end_test, get_config
from automon.functions_to_monitor import set_H
from tests.regression.regression_test_utils import test_func_slack_sync_variations
import numpy as np

regression_test_files_folder = "./regression_test_files_automon/"


def test_func(func_name, NodeClass, data_generator, conf, test_folder):
    test_func_slack_sync_variations(CoordinatorAutoMon, func_name, NodeClass, data_generator, conf, test_folder, regression_test_files_folder)


if __name__ == "__main__":
    try:
        test_folder = start_test("automon_regression")

        conf = get_config(num_nodes=10, num_iterations=200, sliding_window_size=100, d=10, error_bound=0.05,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, domain=(0, 1))
        # Read data from test file
        data_generator = DataGeneratorEntropy(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                              data_file_name=regression_test_files_folder + "data_file_entropy.txt", d=conf["d"])
        test_func("entropy", NodeEntropyAutoMon, data_generator, conf, test_folder)


        conf = get_config(num_nodes=10, num_iterations=100, sliding_window_size=5, d=2, error_bound=2,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        # Read data from test file
        data_generator = DataGeneratorVariance(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                               data_file_name=regression_test_files_folder + "data_file_variance.txt")
        test_func("variance", NodeVarianceAutoMon, data_generator, conf, test_folder)


        conf = get_config(num_nodes=10, num_iterations=100, sliding_window_size=5, d=4, error_bound=0.5,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        # Read data from test file
        data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                   data_file_name=regression_test_files_folder + "data_file_inner_product.txt", d=conf["d"])
        test_func("inner_product", NodeInnerProductAutoMon, data_generator, conf, test_folder)


        conf = get_config(num_nodes=10, num_iterations=150, sliding_window_size=100, d=20, error_bound=0.05,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, domain=(0, 1))
        # Read data from test file
        data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                    data_file_name=regression_test_files_folder + "data_file_kld.txt", d=conf["d"])
        test_func("kld", NodeKLDAutoMon, data_generator, conf, test_folder)


        conf = get_config(num_nodes=10, num_iterations=1000, sliding_window_size=5, d=10, error_bound=2,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        # Read data from test file
        data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                data_file_name=regression_test_files_folder + "data_file_quadratic.txt", d=conf["d"])
        H = np.loadtxt(regression_test_files_folder + 'H_matrix.txt', dtype=np.float32)
        set_H(conf["d"], H)
        test_func("quadratic", NodeQuadraticAutoMon, data_generator, conf, test_folder)


        conf = get_config(num_nodes=3, num_iterations=500, sliding_window_size=10, d=2, error_bound=10,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          domain_type=DomainType.Relative.value, neighborhood_size=0.5)
        # Read data from test file
        data_generator = DataGeneratorRozenbrock(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                 data_file_name=regression_test_files_folder + "data_file_rozenbrock.txt", d=conf["d"])
        test_func("rozenbrock", NodeRozenbrockAutoMon, data_generator, conf, test_folder)

    finally:
        end_test()
