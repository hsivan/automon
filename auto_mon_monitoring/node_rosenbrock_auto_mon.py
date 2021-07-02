import numpy as np
from auto_mon_monitoring.node_common_auto_mon import NodeCommonAutoMon
from functions_to_monitor import func_rozenbrock


class NodeRozenbrockAutoMon(NodeCommonAutoMon):
    
    def __init__(self, idx=0, local_vec_len=2):
        NodeCommonAutoMon.__init__(self, idx, initial_x0=np.zeros(local_vec_len))
        self.func_to_monitor = func_rozenbrock
