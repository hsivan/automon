from automon.cb.cb_cosine_similarity_node import CbCosineSimilarityNode
from automon.cb.cb_inner_product_node import CbInnerProductNode
from automon.cb.cb_coordinator import CbCoordinator
from automon.common_coordinator import SlackType, SyncType
from test_utils.data_generator import DataGeneratorCosineSimilarity, DataGeneratorInnerProduct
from test_utils.functions_to_monitor import func_cosine_similarity, func_inner_product
from test_utils.test_utils import start_test, end_test, get_config
from tests.regression.regression_test_utils import test_func_slack_sync_variations

regression_test_files_folder = "./regression_test_files_cb/"


def test_func(func_name, NodeClass, data_generator, conf, test_folder, func_to_monitor):
    test_func_slack_sync_variations(CbCoordinator, func_name, NodeClass, data_generator, conf, test_folder, regression_test_files_folder, func_to_monitor)


if __name__ == "__main__":
    try:
        test_folder = start_test("cb_regression")

        conf = get_config(num_nodes=10, num_iterations=100, sliding_window_size=5, d=10,
                          error_bound=0.5, slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        # Read data from test file
        data_generator = DataGeneratorCosineSimilarity(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                       data_file_name=regression_test_files_folder + "data_file_cosine_similarity.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        test_func("cosine_similarity", CbCosineSimilarityNode, data_generator, conf, test_folder, func_cosine_similarity)


        conf = get_config(num_nodes=10, num_iterations=100, sliding_window_size=5, d=4,
                          error_bound=0.5, slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        # Read data from test file
        data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                   data_file_name=regression_test_files_folder + "data_file_inner_product.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        test_func("inner_product", CbInnerProductNode, data_generator, conf, test_folder, func_inner_product)

    finally:
        end_test()
