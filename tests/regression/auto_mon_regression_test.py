from automon.automon.node_automon import NodeCommonAutoMon
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.coordinator_common import SlackType, SyncType
from test_utils.data_generator import DataGeneratorEntropy, DataGeneratorVariance, DataGeneratorInnerProduct, \
    DataGeneratorKldAirQuality, DataGeneratorQuadratic, DataGeneratorRozenbrock
from test_utils.test_utils import start_test, end_test, get_config
from test_utils.functions_to_monitor import set_H, func_variance, func_entropy, func_inner_product, func_kld, func_quadratic, func_rozenbrock
from tests.regression.regression_test_utils import test_func_slack_sync_variations
import numpy as np

regression_test_files_folder = "./regression_test_files_automon/"


def test_func(func_name, NodeClass, data_generator, conf, test_folder, func_to_monitor, max_f_val=np.inf, min_f_val=-np.inf):
    test_func_slack_sync_variations(CoordinatorAutoMon, func_name, NodeClass, data_generator, conf, test_folder, regression_test_files_folder, func_to_monitor, max_f_val, min_f_val)


if __name__ == "__main__":
    try:
        test_folder = start_test("automon_regression")

        conf = get_config(num_nodes=10, num_iterations=200, sliding_window_size=100, d=10, error_bound=0.05,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, domain=(0, 1))
        # Read data from test file
        data_generator = DataGeneratorEntropy(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                              data_file_name=regression_test_files_folder + "data_file_entropy.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        test_func("entropy", NodeCommonAutoMon, data_generator, conf, test_folder, func_entropy, max_f_val=func_entropy(np.ones(conf["d"], dtype=np.float) / conf["d"]), min_f_val=0.0)


        conf = get_config(num_nodes=10, num_iterations=100, sliding_window_size=5, d=2, error_bound=2,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        # Read data from test file
        data_generator = DataGeneratorVariance(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                               data_file_name=regression_test_files_folder + "data_file_variance.txt", sliding_window_size=conf["sliding_window_size"])
        test_func("variance", NodeCommonAutoMon, data_generator, conf, test_folder, func_variance, min_f_val=0.0)


        conf = get_config(num_nodes=10, num_iterations=100, sliding_window_size=5, d=4, error_bound=0.5,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        # Read data from test file
        data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                   data_file_name=regression_test_files_folder + "data_file_inner_product.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        test_func("inner_product", NodeCommonAutoMon, data_generator, conf, test_folder, func_inner_product)


        conf = get_config(num_nodes=10, num_iterations=150, sliding_window_size=100, d=20, error_bound=0.05,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, domain=(0, 1))
        # Read data from test file
        data_generator = DataGeneratorKldAirQuality(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                    data_file_name=regression_test_files_folder + "data_file_kld.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        test_func("kld", NodeCommonAutoMon, data_generator, conf, test_folder, func_kld)


        conf = get_config(num_nodes=10, num_iterations=1000, sliding_window_size=5, d=10, error_bound=2,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        # Read data from test file
        data_generator = DataGeneratorQuadratic(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                data_file_name=regression_test_files_folder + "data_file_quadratic.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        H = np.loadtxt(regression_test_files_folder + 'H_matrix.txt', dtype=np.float32)
        set_H(conf["d"], H)
        test_func("quadratic", NodeCommonAutoMon, data_generator, conf, test_folder, func_quadratic)


        conf = get_config(num_nodes=3, num_iterations=500, sliding_window_size=10, d=2, error_bound=10,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, neighborhood_size=0.5)
        # Read data from test file
        data_generator = DataGeneratorRozenbrock(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                 data_file_name=regression_test_files_folder + "data_file_rozenbrock.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        test_func("rozenbrock", NodeCommonAutoMon, data_generator, conf, test_folder, func_rozenbrock)

    finally:
        end_test()
