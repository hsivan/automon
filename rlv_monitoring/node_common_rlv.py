from coordinators.coordinator_common import ViolationOrigin
from fifo import Fifo
import numpy as np
import logging
from timeit import default_timer as timer


class NodeCommonRLV:
    
    def __init__(self, idx=0, initial_x0=None, max_f_val=np.inf, min_f_val=-np.inf):
        self.l_thresh = 0
        self.u_thresh = 0
        self.slack = 0
        self.coordinator = None  # For verifier node there is no coordinator (no call to set_coordinator)
        self.idx = idx
        self.domain = None
        self.b_before_first_sync = True
        self.x0 = initial_x0.copy()  # Global probability vector
        self.x = initial_x0.copy()  # Current local probability vector
        self.x0_local = initial_x0.copy()  # Local probability vector at the time of the last sync
        self.max_f_val = max_f_val  # The maximum value of the monitored function (if known, otherwise inf)
        self.min_f_val = min_f_val  # The minimum value of the monitored function (if known, otherwise -inf)

        self.inside_bounds_evaluation_accumulated_time = 0  # Count only data points inside domain, when f is evaluated
        self.inside_bounds_evaluation_counter = 0  # Count only data points inside domain, when f is evaluated
    
    def _get_point_to_check(self):
        point_to_check = self.x - self.slack
        return point_to_check

    def _inside_domain(self, x):
        # Check if the point is inside the domain.
        # If the domain is None it contains the entire sub-space and therefore, the point is always inside the domain.
        # Otherwise, the domain is a list of tuples [(min_domain_x_0,max_domain_x_0),(min_domain_x_1,max_domain_x_1),...].
        if self.domain is None:
            return True

        if not np.all(x >= np.array([min_domain for (min_domain, max_domain) in self.domain])):
            return False
        if not np.all(x <= np.array([max_domain for (min_domain, max_domain) in self.domain])):
            return False

        return True

    def _inside_bounds(self, x):
        start = timer()

        f_at_x = self.func_to_monitor(x)
        b_inside_bounds = self.l_thresh <= f_at_x <= self.u_thresh

        end = timer()
        self.inside_bounds_evaluation_accumulated_time += end - start
        self.inside_bounds_evaluation_counter += 1

        return b_inside_bounds

    def _report_violation(self, violation_origin):
        if self.coordinator is not None:
            self.coordinator.notify_violation(self.idx, self.x, violation_origin)

    # This function is called internally from set_new_data_point() to verify the new local vector.
    # It is also called by the coordinator that uses dummy node as part of the lazy sync procedure, to verify if it was
    # able to resolve violations with subset of the nodes.
    def inside_effective_safe_zone(self, x):
        if self.b_before_first_sync:  # No sync was called just yet. Automatically report violation to trigger sync.
            self._report_violation(ViolationOrigin.SafeZone)
            return False

        if not self._inside_domain(x):
            logging.info("Node " + str(self.idx) + " local vector is outside the domain")
            self._report_violation(ViolationOrigin.Domain)
            return False

        if not self._inside_bounds(x):
            self._report_violation(ViolationOrigin.SafeZone)
            return False

        return True
    
    def set_new_data_point(self, data_point):
        # Use the data point to update the local vector.
        # If needed, report violation to coordinator.
        self.x = data_point
        x = self._get_point_to_check()
        res = self.inside_effective_safe_zone(x)
        return res
    
    def sync(self, x0, slack, l_thresh, u_thresh, domain):
        self.b_before_first_sync = False
        self.x0 = x0.copy()
        self.slack = slack
        self.x0_local = self.x.copy()  # Fix local vector at sync

        self.l_thresh = l_thresh
        self.u_thresh = u_thresh
        self.domain = domain
            
    def set_coordinator(self, coordinator):
        self.coordinator = coordinator
    
    def get_local_vector(self):
        return self.x

    def dump_stats(self):
        logging.info("Node " + str(self.idx) + " inside_bounds evaluation counter " + str(self.inside_bounds_evaluation_counter))
        if self.inside_bounds_evaluation_counter > 0:
            logging.info("Node " + str(self.idx) + " Avg inside_bounds evaluation time " + str(self.inside_bounds_evaluation_accumulated_time / self.inside_bounds_evaluation_counter))
