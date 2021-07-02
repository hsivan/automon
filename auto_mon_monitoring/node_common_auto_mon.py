from coordinators.coordinator_auto_mon import DcType
from coordinators.coordinator_common import ViolationOrigin
from fifo import Fifo
import logging
import numpy
from timeit import default_timer as timer


class NodeCommonAutoMon:
    
    def __init__(self, idx=0, initial_x0=None, max_f_val=numpy.inf, min_f_val=-numpy.inf):
        self.l_thresh = 0
        self.u_thresh = 0
        self.slack = 0
        self.coordinator = None  # For verifier node there is no coordinator (no call to set_coordinator)
        self.idx = idx
        self.domain = None
        self.g_func = None
        self.h_func = None
        self.b_before_first_sync = True
        self.x0 = initial_x0.copy()  # Global probability vector
        self.x = initial_x0.copy()  # Current local probability vector
        self.x0_local = initial_x0.copy()  # Local probability vector at the time of the last sync
        self.max_f_val = max_f_val  # The maximum value of the monitored function (if known, otherwise inf)
        self.min_f_val = min_f_val  # The minimum value of the monitored function (if known, otherwise -inf)

        self.full_sync_accumulated_time = 0
        self.full_sync_accumulated_time_square = 0
        self.full_sync_counter = 0
        self.inside_safe_zone_evaluation_accumulated_time = 0
        self.inside_safe_zone_evaluation_accumulated_time_square = 0
        self.inside_safe_zone_evaluation_counter = 0
        self.inside_bounds_evaluation_accumulated_time = 0
        self.inside_bounds_evaluation_accumulated_time_square = 0
        self.inside_bounds_evaluation_counter = 0
        self.data_update_accumulated_time = 0
        self.data_update_accumulated_time_square = 0
        self.data_update_counter = 0
    
    def _get_point_to_check(self):
        point_to_check = self.x - self.slack
        return point_to_check
    
    def _below_safe_zone_upper_bound(self, x):
        # Check if the u_thresh is above the maximum value of the monitored function.
        # In that special case, every point is under the safe zone upper bound.
        if self.u_thresh >= self.max_f_val:
            b_under_upper_threshold = True
            return b_under_upper_threshold
        
        # The hyperplane equation of the tangent plane to h at the point x0, plus the upper threshold is:
        # z(x) = h(x0) + grad_h(x0)*(x-x0) + u_thresh.
        # Need to check if:
        # g(x) <= z(x).
        # If true - f(x) is below the upper threshold (inside the safe zone).
        # Otherwise, f(x) is above the upper threshold (outside the safe zone).
        z_x = self.h_func_at_x0 + self.h_func_grad_at_x0 @ (x - self.x0) + self.u_thresh
        g_x = self.g_func(x)
        if g_x <= z_x:
            return True
        return False
    
    def _above_safe_zone_lower_bound(self, x):
        # Check if the l_thresh is bellow the minimum value of the monitored function.
        # In that special case, every point is below the safe zone lower bound.
        if self.l_thresh <= self.min_f_val:
            b_above_lower_threshold = True
            return b_above_lower_threshold
        
        # The hyperplane equation of the tangent plane to g at the point x0, minus the lower threshold is:
        # z(x) = g(x0) + grad_g(x0)*(x-x0) - l_thresh.
        # Need to check if:
        # h(x) <= z(x).
        # If true - f(x) is above the lower threshold (inside the safe zone).
        # Otherwise, f(x) is below the lower threshold (outside the safe zone).
        z_x = self.g_func_at_x0 + self.g_func_grad_at_x0 @ (x - self.x0) - self.l_thresh
        h_x = self.h_func(x)
        if h_x <= z_x:
            return True
        return False
    
    def _inside_domain(self, x):
        # Check if the point is inside the domain.
        # If the domain is None it contains the entire sub-space and therefore, the point is always inside the domain.
        # Otherwise, the domain is a list of tuples [(min_domain_x_0,max_domain_x_0),(min_domain_x_1,max_domain_x_1),...].
        if self.domain is None:
            return True

        if not numpy.all(x >= numpy.array([min_domain for (min_domain, max_domain) in self.domain])):
            return False
        if not numpy.all(x <= numpy.array([max_domain for (min_domain, max_domain) in self.domain])):
            return False
        
        return True

    def _inside_safe_zone(self, x):
        start = timer()

        b_inside_safe_zone = self._above_safe_zone_lower_bound(x) and self._below_safe_zone_upper_bound(x)

        end = timer()
        self.inside_safe_zone_evaluation_accumulated_time += end - start
        self.inside_safe_zone_evaluation_accumulated_time_square += (end - start)**2
        self.inside_safe_zone_evaluation_counter += 1

        return b_inside_safe_zone

    def _inside_bounds(self, x):
        start = timer()

        f_at_x = self.func_to_monitor(x)
        b_inside_bounds = self.l_thresh <= f_at_x <= self.u_thresh

        end = timer()
        self.inside_bounds_evaluation_accumulated_time += end - start
        self.inside_bounds_evaluation_accumulated_time_square += (end - start)**2
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

        if not self._inside_safe_zone(x):
            self._report_violation(ViolationOrigin.SafeZone)
            return False

        if not self._inside_bounds(x):
            # There is a problem with the constraints. Coordinator must be informed and perform full sync.
            self._report_violation(ViolationOrigin.FaultySafeZone)
            return False

        return True

    def set_new_data_point(self, data_point):
        start = timer()

        # Use the data point to update the local vector.
        # If needed, report violation to coordinator.
        # The "heavy" computations in this function are the calls to g_func(x), h_func(x), and func_to_monitor(x).
        self.x = data_point
        x = self._get_point_to_check()
        res = self.inside_effective_safe_zone(x)

        end = timer()
        self.data_update_accumulated_time += end - start
        self.data_update_accumulated_time_square += (end - start)**2
        self.data_update_counter += 1

        return res

    def _func_dc_operator(self, x, dc_argument, x0):
        if len(x.shape) < 2:
            return (x - x0) @ dc_argument @ (x - x0).T
        # Only for figures.
        return numpy.sum(((x - x0) @ dc_argument) * (x - x0), axis=1)

    def _get_dc(self, dc_argument, x0):
        if len(dc_argument.shape) == 0:
            # In case dc_argument is float (extreme eigenvalue) convert to matrix to get unified code for ADCD-E and ADCD-X
            dc_argument = dc_argument * numpy.eye(x0.shape[0])
        g_func = lambda x: self.func_to_monitor(x) - 0.5 * self._func_dc_operator(x, dc_argument, x0)
        h_func = lambda x: -0.5 * self._func_dc_operator(x, dc_argument, x0)
        return g_func, h_func
    
    def sync(self, x0, slack, l_thresh=None, u_thresh=None, domain=None, dc_type=None, dc_argument=None, g_func_grad_at_x0=None, h_func_grad_at_x0=None):
        self.b_before_first_sync = False
        self.x0 = x0.copy()
        self.slack = slack
        self.x0_local = self.x.copy()  # Fix local vector at sync

        if l_thresh is not None:
            self.l_thresh = l_thresh
            self.u_thresh = u_thresh
            self.domain = domain

            start = timer()

            # This sync() is called as a result of full sync and therefore the coordinator sent update for the constraints.
            # Compute everything that can be computed now to save computations every call to set_new_data_point().
            g_func, h_func = self._get_dc(dc_argument, x0)
            if dc_type == DcType.Convex:
                self.g_func = g_func
                self.h_func = h_func
                self.g_func_grad_at_x0 = g_func_grad_at_x0  # The tangent line of g at the point x0
                self.h_func_grad_at_x0 = h_func_grad_at_x0  # The tangent line of h at the point x0
            else:
                # The dc type is DcType.Concave.
                # To get unified code for convex diff and concave diff constraints, we convert concave diff to be convex diff.
                self.g_func = lambda x: -h_func(x)
                self.h_func = lambda x: -g_func(x)
                self.g_func_grad_at_x0 = -h_func_grad_at_x0  # The tangent line of g at the point x0
                self.h_func_grad_at_x0 = -g_func_grad_at_x0  # The tangent line of h at the point x0
            self.g_func_at_x0 = self.g_func(self.x0).copy()
            self.h_func_at_x0 = self.h_func(self.x0).copy()

            end = timer()
            self.full_sync_accumulated_time += end - start
            self.full_sync_accumulated_time_square += (end - start)**2
            self.full_sync_counter += 1

    def set_coordinator(self, coordinator):
        self.coordinator = coordinator
    
    def get_local_vector(self):
        return self.x

    def _log_time_mean_and_std(self, counter, accumulated_time, accumulated_time_square, timer_title):
        logging.info("Node " + str(self.idx) + " " + timer_title + " counter " + str(counter))
        if counter > 0:
            mean = accumulated_time / counter
            var = (accumulated_time_square / counter) - mean ** 2
            std = numpy.sqrt(var)
            logging.info("Node " + str(self.idx) + " Avg " + timer_title + " time " + str(mean))
            logging.info("Node " + str(self.idx) + " Std " + timer_title + " time " + str(std))

    def dump_stats(self):
        self._log_time_mean_and_std(self.full_sync_counter,
                                    self.full_sync_accumulated_time,
                                    self.full_sync_accumulated_time_square,
                                    "full sync")
        self._log_time_mean_and_std(self.inside_safe_zone_evaluation_counter,
                                    self.inside_safe_zone_evaluation_accumulated_time,
                                    self.inside_safe_zone_evaluation_accumulated_time_square,
                                    "inside_safe_zone evaluation")
        self._log_time_mean_and_std(self.inside_bounds_evaluation_counter,
                                    self.inside_bounds_evaluation_accumulated_time,
                                    self.inside_bounds_evaluation_accumulated_time_square,
                                    "inside_bounds evaluation")
        self._log_time_mean_and_std(self.data_update_counter,
                                    self.data_update_accumulated_time,
                                    self.data_update_accumulated_time_square,
                                    "data update")
