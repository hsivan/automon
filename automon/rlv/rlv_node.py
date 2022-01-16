from automon.common_node import CommonNode
from automon.common_messages import ViolationOrigin, parse_message_sync
import numpy as np
from timeit import default_timer as timer


class RlvNode(CommonNode):
    
    def __init__(self, idx, func_to_monitor, d=1, max_f_val=np.inf, min_f_val=-np.inf, domain=None):
        CommonNode.__init__(self, idx, func_to_monitor, d, domain, max_f_val, min_f_val, node_name="RLV")
        self._init()

    def _init(self):
        super()._init()
        self.inside_bounds_evaluation_counter = 0  # Count only data points inside domain, when f is evaluated
        self.inside_bounds_evaluation_accumulated_time = 0
        self.inside_bounds_evaluation_accumulated_time_square = 0

    def _inside_bounds(self, x):
        start = timer()

        f_at_x = self.func_to_monitor(x)
        b_inside_bounds = self.l_thresh <= f_at_x <= self.u_thresh

        end = timer()
        self.inside_bounds_evaluation_accumulated_time += end - start
        self.inside_bounds_evaluation_accumulated_time_square += (end - start) ** 2
        self.inside_bounds_evaluation_counter += 1

        return b_inside_bounds

    def inside_effective_safe_zone(self, x):
        if self.b_before_first_sync:  # No sync was called just yet. Automatically report violation to trigger sync.
            self._report_violation(ViolationOrigin.SafeZone)
            return False

        if not self._inside_domain(x):
            self._report_violation(ViolationOrigin.Domain)
            return False

        if not self._inside_bounds(x):
            self._report_violation(ViolationOrigin.SafeZone)
            return False

        return True
    
    def sync(self, x0, slack, l_thresh, u_thresh):
        super()._sync_common(x0, slack, l_thresh, u_thresh)

    # Override
    def dump_stats(self, test_folder):
        self._log_time_mean_and_std(self.inside_bounds_evaluation_counter,
                                    self.inside_bounds_evaluation_accumulated_time,
                                    self.inside_bounds_evaluation_accumulated_time_square,
                                    "inside_bounds evaluation")
        super().dump_stats(test_folder)

    # Override
    def _handle_sync_message(self, payload):
        constraint_version, x0, slack, l_thresh, u_thresh = parse_message_sync(payload, self.x0.shape[0])
        self.constraint_version = constraint_version
        self.sync(x0, slack, l_thresh, u_thresh)
