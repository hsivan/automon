from automon.common_coordinator import CommonCoordinator, SlackType, SyncType
from automon.rlv.rlv_node import RlvNode
import numpy as np


class RlvCoordinator(CommonCoordinator):
    
    def __init__(self, num_nodes, func_to_monitor, error_bound=2, slack_type=SlackType.Drift, sync_type=SyncType.Eager,
                 lazy_sync_max_S=0.5, d=1, max_f_val=np.inf, min_f_val=-np.inf, domain=None):
        # Create a dummy node for the coordinator that uses it in the process of resolving violations.
        # In simulations the verifier's local vector is the global vector (updated by the test manager), and it is used for collecting statistics.
        verifier = RlvNode(-1, func_to_monitor, d, max_f_val, min_f_val, domain)
        # To force using RLV with slack, use slack_type and sync_type instead of the const values below
        CommonCoordinator.__init__(self, verifier, num_nodes, error_bound, SlackType.NoSlack, SyncType.Eager, lazy_sync_max_S,
                                   b_violation_strict=False, coordinator_name="RLV")
