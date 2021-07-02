import numpy as np
from auto_mon_monitoring.node_common_auto_mon import NodeCommonAutoMon
from functions_to_monitor import func_variance


class NodeVarianceAutoMon(NodeCommonAutoMon):
    
    def __init__(self, idx=0, local_vec_len=2):
        NodeCommonAutoMon.__init__(self, idx, initial_x0=np.zeros(local_vec_len), min_f_val=0)
        assert(local_vec_len == 2)  # The local vector is the first and second momentum
        self.func_to_monitor = func_variance
