from automon.common_node import CommonNode
from automon.common_messages import ViolationOrigin, parse_message_sync
import numpy as np
from timeit import default_timer as timer


class CbCommonNode(CommonNode):
    
    def __init__(self, idx=0, d=1, max_f_val=np.inf, min_f_val=-np.inf, domain=None, func_to_monitor=None):
        CommonNode.__init__(self, idx, func_to_monitor, d, domain, max_f_val, min_f_val, node_name="CB")
        self._init()

    def _init(self):
        super()._init()
        self.inside_safe_zone_evaluation_counter = 0  # Count only data points inside domain, when g,h are evaluated
        self.inside_safe_zone_evaluation_accumulated_time = 0
        self.inside_safe_zone_evaluation_accumulated_time_square = 0

    def _func_h(self, X, threshold):
        raise NotImplementedError("To be implemented by inherent class")

    def _func_g(self, X, threshold):
        raise NotImplementedError("To be implemented by inherent class")

    def _func_h_grad(self, X, threshold):
        raise NotImplementedError("To be implemented by inherent class")

    def _func_g_grad(self, X, threshold):
        raise NotImplementedError("To be implemented by inherent class")

    def _below_safe_zone_upper_bound(self, x):
        # Check if the u_thresh is above the maximum value of the monitored function.
        # In that special case, every point is under the safe zone upper bound.
        if self.u_thresh >= self.max_f_val:
            b_under_upper_threshold = True
            return b_under_upper_threshold

        # The hyperplane equation of the tangent plane to g at the point x0 is:
        # z(x) = g(x0) + grad_g(x0)*(x-x0).
        # Need to check if:
        # h(x) <= z(x).
        # If true - f(x) is above the lower threshold (inside the safe zone).
        # Otherwise, f(x) is below the lower threshold (outside the safe zone).
        z_x = self._func_g(self.x0, self.u_thresh) + self._func_g_grad(self.x0, self.u_thresh) @ (x - self.x0)
        h_x = self._func_h(x, self.u_thresh)
        if h_x <= z_x:
            return True
        return False

    def _above_safe_zone_lower_bound(self, x):
        # Check if the l_thresh is bellow the minimum value of the monitored function.
        # In that special case, every point is below the safe zone lower bound.
        if self.l_thresh <= self.min_f_val:
            b_above_lower_threshold = True
            return b_above_lower_threshold

        # The hyperplane equation of the tangent plane to h at the point x0 is:
        # z(x) = h(x0) + grad_h(x0)*(x-x0).
        # Need to check if:
        # g(x) <= z(x).
        # If true - f(x) is below the upper threshold (inside the safe zone).
        # Otherwise, f(x) is above the upper threshold (outside the safe zone).
        z_x = self._func_h(self.x0, self.l_thresh) + self._func_h_grad(self.x0, self.l_thresh) @ (x - self.x0)
        g_x = self._func_g(x, self.l_thresh)
        if g_x <= z_x:
            return True
        return False

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
