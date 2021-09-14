from automon.node_common import NodeCommon
from automon.messages_common import ViolationOrigin, parse_message_sync
from timeit import default_timer as timer


class NodeCommonGM(NodeCommon):
    
    def __init__(self, idx=0, x0_len=1, domain=None, func_to_monitor=None):
        NodeCommon.__init__(self, idx, func_to_monitor, x0_len, domain)
        self.node_name = "GM"
        self._init()

    def _init(self):
        super()._init()
        self.q = None
        self.inside_safe_zone_evaluation_counter = 0  # Count only data points inside domain, when safe zone is evaluated
        self.inside_safe_zone_evaluation_accumulated_time = 0
        self.inside_safe_zone_evaluation_accumulated_time_square = 0

    def _below_safe_zone_upper_bound(self, x):
        raise NotImplementedError("To be implemented by inherent class")

    def _above_safe_zone_lower_bound(self, x):
        raise NotImplementedError("To be implemented by inherent class")

    def _calc_q(self):
        raise NotImplementedError("To be implemented by inherent class")

    def _inside_safe_zone(self, x):
        start = timer()

        b_inside_safe_zone = self._above_safe_zone_lower_bound(x) and self._below_safe_zone_upper_bound(x)

        end = timer()
        self.inside_safe_zone_evaluation_accumulated_time += end - start
        self.inside_safe_zone_evaluation_accumulated_time_square += (end - start) ** 2
        self.inside_safe_zone_evaluation_counter += 1

        return b_inside_safe_zone

    def inside_effective_safe_zone(self, x):
        if self.b_before_first_sync:  # No sync was called just yet. Automatically report violation to trigger sync.
            self._report_violation(ViolationOrigin.SafeZone)
            return False

        if not self._inside_domain(x):
            self._report_violation(ViolationOrigin.Domain)
            return False

        if not self._inside_safe_zone(x):
            self._report_violation(ViolationOrigin.SafeZone)
            return False

        return True
    
    def sync(self, x0, slack, l_thresh, u_thresh):
        super()._sync_common(x0, slack, l_thresh, u_thresh)
        # The point q should be computed now, after the update of x0
        self.q = self._calc_q()

    # Override
    def dump_stats(self, test_folder):
        self._log_time_mean_and_std(self.inside_safe_zone_evaluation_counter,
                                    self.inside_safe_zone_evaluation_accumulated_time,
                                    self.inside_safe_zone_evaluation_accumulated_time_square,
                                    "inside_safe_zone evaluation")
        super().dump_stats(test_folder)

    # Override
    def _handle_sync_message(self, payload):
        constraint_version, x0, slack, l_thresh, u_thresh = parse_message_sync(payload, self.x0.shape[0])
        self.constraint_version = constraint_version
        self.sync(x0, slack, l_thresh, u_thresh)
