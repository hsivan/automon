from automon.common_coordinator import CommonCoordinator, SlackType, SyncType


class GmCoordinator(CommonCoordinator):
    
    def __init__(self, NodeClass, num_nodes, func_to_monitor, error_bound=2, slack_type=SlackType.Drift, sync_type=SyncType.Eager,
                 lazy_sync_max_S=0.5, d=1, domain=None):
        # Create a dummy node for the coordinator that uses it in the process of resolving violations.
        # In simulations the verifier's local vector is the global vector (updated by the test manager), and it is used for collecting statistics.
        verifier = NodeClass(idx=-1, func_to_monitor=func_to_monitor, d=d, domain=domain)
        CommonCoordinator.__init__(self, verifier, num_nodes, error_bound, slack_type, sync_type, lazy_sync_max_S,
                                   b_violation_strict=True, coordinator_name="GM")
