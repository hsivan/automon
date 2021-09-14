import logging
from automon.coordinator_common import CoordinatorCommon, SlackType, SyncType


class CoordinatorRLV(CoordinatorCommon):
    
    def __init__(self, verifier, func_to_monitor, num_nodes, error_bound=2,
                 slack_type=SlackType.Drift, sync_type=SyncType.Eager, lazy_sync_max_S=0.5, domain=None):
        # To force using RLV with slack remove the following two lines
        slack_type = SlackType.NoSlack
        sync_type = SyncType.Eager
        CoordinatorCommon.__init__(self, verifier, func_to_monitor, num_nodes, error_bound, slack_type, sync_type, lazy_sync_max_S)
        logging.info("CoordinatorRLV initialization: domain " + str(domain))
        self.coordinator_name = "RLV"
        self.b_violation_strict = False
        self.domain = [domain] * self.x0_len if domain is not None else None
