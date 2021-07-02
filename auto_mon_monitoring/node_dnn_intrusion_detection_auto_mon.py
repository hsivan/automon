from auto_mon_monitoring.node_common_auto_mon import NodeCommonAutoMon
import numpy as np
from functions_to_monitor import func_dnn_intrusion_detection


class NodeDnnIntrusionDetectionAutoMon(NodeCommonAutoMon):
    
    def __init__(self, idx=0, local_vec_len=41):
        NodeCommonAutoMon.__init__(self, idx, initial_x0=np.zeros(local_vec_len, dtype=np.float))
        self.func_to_monitor = func_dnn_intrusion_detection
