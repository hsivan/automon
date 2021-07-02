from rlv_monitoring.node_common_rlv import NodeCommonRLV
import numpy as np
from functions_to_monitor import func_dnn_intrusion_detection


class NodeDnnIntrusionDetectionRLV(NodeCommonRLV):

    def __init__(self, idx=0, local_vec_len=41):
        NodeCommonRLV.__init__(self, idx, initial_x0=np.zeros(local_vec_len))
        self.func_to_monitor = func_dnn_intrusion_detection
