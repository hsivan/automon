from automon.coordinator_common import CoordinatorCommon, SlackType, SyncType


class CoordinatorRLV(CoordinatorCommon):
    
    def __init__(self, verifier, num_nodes, error_bound=2, slack_type=SlackType.Drift, sync_type=SyncType.Eager,
                 lazy_sync_max_S=0.5):
        # To force using RLV with slack use slack_type and sync_type instead of the const values
        CoordinatorCommon.__init__(self, verifier, num_nodes, error_bound, SlackType.NoSlack, SyncType.Eager, lazy_sync_max_S,
                                   b_violation_strict=False, coordinator_name="RLV")
