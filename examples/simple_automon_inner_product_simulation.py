import sys
from importlib import reload
import logging
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon_utils.data_generator import DataGeneratorInnerProduct
from automon_utils.test_utils import run_test, get_config
from automon.automon.node_common_automon import NodeCommonAutoMon
from automon_utils.functions_to_monitor import func_inner_product

if __name__ == "__main__":
    reload(logging)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    conf = get_config(num_iterations=1020, sliding_window_size=20, d=40, error_bound=0.3)
    data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], sliding_window_size=conf["sliding_window_size"])

    nodes = [NodeCommonAutoMon(idx, x0_len=conf["d"], func_to_monitor=func_inner_product) for idx in range(conf["num_nodes"])]

    verifier = NodeCommonAutoMon(idx=-1, x0_len=conf["d"], func_to_monitor=func_inner_product)

    coordinator = CoordinatorAutoMon(verifier, conf["num_nodes"], error_bound=conf["error_bound"])

    logging.info("\n###################### Start inner product AutoMon test ######################")
    run_test(data_generator, coordinator, nodes, None)
