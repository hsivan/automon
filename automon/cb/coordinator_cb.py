import logging
from automon.coordinator_common import CoordinatorCommon, SlackType, SyncType
    

class CoordinatorCB(CoordinatorCommon):
    
    def __init__(self, verifier, func_to_monitor, num_nodes, error_bound=2,
                 slack_type=SlackType.Drift, sync_type=SyncType.Eager, lazy_sync_max_S=0.5, domain=None):
        
        CoordinatorCommon.__init__(self, verifier, func_to_monitor, num_nodes, error_bound, slack_type, sync_type, lazy_sync_max_S)
        logging.info("CoordinatorCB initialization: domain " + str(domain))
        self.coordinator_name = "CB"
        self.b_violation_strict = True
        self.domain = domain
