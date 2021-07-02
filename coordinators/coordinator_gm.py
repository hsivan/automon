import numpy as np
import logging
from coordinators.coordinator_common import CoordinatorCommon, SlackType, SyncType


class CoordinatorGM(CoordinatorCommon):
    
    def __init__(self, verifier, func_to_monitor, x0_len, num_nodes, error_bound=2,
                 slack_type=SlackType.Drift, sync_type=SyncType.Eager, lazy_sync_max_S=0.5, domain=None):
        CoordinatorCommon.__init__(self, verifier, func_to_monitor, x0_len, num_nodes, error_bound, slack_type, sync_type, lazy_sync_max_S)
        logging.info("CoordinatorGM initialization")
        self.coordinator_name = "GM"
        self.b_violation_strict = True
        self.domain = [domain] * x0_len if domain is not None else None  # Domain is NOT supported in GM nodes. Only GM Entropy function implementation use it at the moment, but implicitly by checking if local vector values are less that 0.

    # Override - difference in sync
    def _sync_verifier(self):
        # Since verifier.x equals new_x0, no slack is ever needed.
        self.verifier.sync(self.x0, np.zeros_like(self.x0), self.l_thresh, self.u_thresh)
        
    # Override - difference in sync
    def _sync_node(self, node_idx, sync_type="full"):
        node = self.nodes[node_idx]
        node.sync(self.x0, self.nodes_slack[node_idx], self.l_thresh, self.u_thresh)
