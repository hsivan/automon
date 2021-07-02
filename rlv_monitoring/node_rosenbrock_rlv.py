import numpy as np
from rlv_monitoring.node_common_rlv import NodeCommonRLV
from functions_to_monitor import func_rozenbrock


class NodeRozenbrockRLV(NodeCommonRLV):
    
    def __init__(self, idx=0, local_vec_len=2):
        NodeCommonRLV.__init__(self, idx, initial_x0=np.zeros(local_vec_len))
        self.func_to_monitor = func_rozenbrock
