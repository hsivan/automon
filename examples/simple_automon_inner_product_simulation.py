import sys
import logging
from automon.auto_mon.coordinator_automon import CoordinatorAutoMon
from test_utils.data_generator import DataGeneratorInnerProduct
from test_utils.test_utils import run_test, get_config
from automon.auto_mon.node_automon import NodeAutoMon
from test_utils.functions_to_monitor import func_inner_product
logging.getLogger('automon').setLevel(logging.INFO)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    conf = get_config(num_iterations=1020, sliding_window_size=20, d=40, error_bound=0.3)
    data_generator = DataGeneratorInnerProduct(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], sliding_window_size=conf["sliding_window_size"])

    nodes = [NodeAutoMon(idx, x0_len=conf["d"], func_to_monitor=func_inner_product) for idx in range(conf["num_nodes"])]

    verifier = NodeAutoMon(idx=-1, x0_len=conf["d"], func_to_monitor=func_inner_product)

    coordinator = CoordinatorAutoMon(verifier, conf["num_nodes"], error_bound=conf["error_bound"])

    run_test(data_generator, coordinator, nodes, None)
