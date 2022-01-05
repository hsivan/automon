from automon.coordinator_common import CoordinatorCommon, SlackType, SyncType
    

class CoordinatorCB(CoordinatorCommon):
    
    def __init__(self, verifier, num_nodes, error_bound=2, slack_type=SlackType.Drift, sync_type=SyncType.Eager,
                 lazy_sync_max_S=0.5):
        CoordinatorCommon.__init__(self, verifier, num_nodes, error_bound, slack_type, sync_type, lazy_sync_max_S,
                                   b_violation_strict=True, coordinator_name="CB")
