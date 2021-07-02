from auto_mon_monitoring.node_common_auto_mon import NodeCommonAutoMon
import numpy as np
from functions_to_monitor import func_entropy


class NodeEntropyAutoMon(NodeCommonAutoMon):

    def __init__(self, idx=0, local_vec_len=2):
        initial_x0 = np.ones(local_vec_len, dtype=np.float) / local_vec_len
        NodeCommonAutoMon.__init__(self, idx, initial_x0=initial_x0, max_f_val=func_entropy(initial_x0), min_f_val=0)
        self.func_to_monitor = func_entropy
