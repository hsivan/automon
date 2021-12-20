import logging
from automon.coordinator_common import CoordinatorCommon, SlackType, SyncType


class CoordinatorGM(CoordinatorCommon):
    
    def __init__(self, verifier, num_nodes, error_bound=2, slack_type=SlackType.Drift, sync_type=SyncType.Eager,
                 lazy_sync_max_S=0.5, domain=None):
        CoordinatorCommon.__init__(self, verifier, num_nodes, error_bound, slack_type, sync_type, lazy_sync_max_S)
        logging.info("CoordinatorGM initialization")
        self.coordinator_name = "GM"
        self.b_violation_strict = True
        self.domain = [domain] * self.x0_len if domain is not None else None  # Domain is NOT supported in GM nodes. Only GM Entropy function implementation use it at the moment, but implicitly by checking if local vector values are less that 0.
