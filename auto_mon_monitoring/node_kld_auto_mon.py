from auto_mon_monitoring.node_common_auto_mon import NodeCommonAutoMon
import numpy as np
from functions_to_monitor import func_kld


class NodeKLDAutoMon(NodeCommonAutoMon):

    def __init__(self, idx=0, local_vec_len=2):
        k = local_vec_len // 2  # Two concatenated frequency vectors
        initial_x0 = np.ones(local_vec_len, dtype=np.float) / k
        NodeCommonAutoMon.__init__(self, idx, initial_x0=initial_x0)
        self.func_to_monitor = func_kld
