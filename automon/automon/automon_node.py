import os
import numpy
from timeit import default_timer as timer
import logging
from automon.common_messages import ViolationOrigin
from automon.automon.automon_messages import DcType, parse_message_sync_automon
from automon.common_node import CommonNode

logging = logging.getLogger(__name__)

# Can set this environment variable in test to force use AutoGrad, even if Jax exists: os.environ['AUTO_DIFFERENTIATION_TOOL'] = 'AutoGrad'/'Jax'
AUTO_DIFFERENTIATION_TOOL = os.getenv('AUTO_DIFFERENTIATION_TOOL')
# Try import Jax if AUTO_DIFFERENTIATION_TOOL is None (automatic decision, try Jax first) or AUTO_DIFFERENTIATION_TOOL is Jax.
# If failed to import Jax or AUTO_DIFFERENTIATION_TOOL is AutoGrad use AutoGrad
if (AUTO_DIFFERENTIATION_TOOL is None) or (AUTO_DIFFERENTIATION_TOOL == "Jax"):
    try:
        import jax
        AUTO_DIFFERENTIATION_TOOL = "Jax"
        from jax import jit
        from jax import grad as jax_grad
        from jax.config import config
        config.update("jax_enable_x64", True)
        config.update("jax_platform_name", 'cpu')

        def grad(fun):
            return jit(jax_grad(fun))
    except Exception as e:
        AUTO_DIFFERENTIATION_TOOL = "AutoGrad"
        from autograd import grad


class AutomonNode(CommonNode):
    
    def __init__(self, idx, func_to_monitor, d=1, max_f_val=numpy.inf, min_f_val=-numpy.inf, domain=None):
        CommonNode.__init__(self, idx, func_to_monitor, d, domain, max_f_val, min_f_val, node_name="AutoMon")
        logging.info("AutoMon node " + str(idx) + " initialization: AUTO_DIFFERENTIATION_TOOL " + AUTO_DIFFERENTIATION_TOOL)
        self.domain_range = domain
        self.func_to_monitor_grad = grad(func_to_monitor)
        self._init()

    def _init(self):
        super()._init()
        self.domain = [self.domain_range] * self.d if self.domain_range is not None else None
        self.g_func = None
        self.h_func = None

        self.full_sync_counter = 0
        self.full_sync_accumulated_time = 0
        self.full_sync_accumulated_time_square = 0
        self.inside_safe_zone_evaluation_counter = 0
        self.inside_safe_zone_evaluation_accumulated_time = 0
        self.inside_safe_zone_evaluation_accumulated_time_square = 0
        self.inside_bounds_evaluation_counter = 0
        self.inside_bounds_evaluation_accumulated_time = 0
        self.inside_bounds_evaluation_accumulated_time_square = 0

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
        # If l_thresh is smaller than min_f_val, than the global vector could not violate the lower bound constraint.
        # Similarly, if u_thresh is larger than max_f_val, than the global vector could not violate the upper bound constraint.
        # Note that x (which is self.x minus the slack) could lead to f_at_x < l_thresh or f_at_x > u_thresh), e.g. in variance,
        # since x is not a "natural" vector, and it should be ignored if we aware that l_thresh < min_f_val or u_thresh > max_f_val.
        b_inside_bounds = (self.l_thresh <= f_at_x or self.l_thresh <= self.min_f_val) and (f_at_x <= self.u_thresh or self.u_thresh >= self.max_f_val)

        end = timer()
        self.inside_bounds_evaluation_accumulated_time += end - start
        self.inside_bounds_evaluation_accumulated_time_square += (end - start)**2
        self.inside_bounds_evaluation_counter += 1

        return b_inside_bounds

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

        if not self._inside_bounds(x):
            # There is a problem with the constraints. Coordinator must be informed and perform full sync.
            self._report_violation(ViolationOrigin.FaultySafeZone)
            return False

        return True

    def _func_dc_operator(self, x, x0):
        if len(x.shape) < 2:
            return (x - x0) @ self.dc_argument @ (x - x0).T
        # Only for figures.
        return numpy.sum(((x - x0) @ self.dc_argument) * (x - x0), axis=1)

    def _get_dc(self, x0):
        g_func = lambda x: self.func_to_monitor(x) - 0.5 * self._func_dc_operator(x, x0)
        h_func = lambda x: -0.5 * self._func_dc_operator(x, x0)
        return g_func, h_func

    def sync(self, x0, slack, l_thresh, u_thresh, neighborhood_size, dc_type, dc_argument):
        super()._sync_common(x0, slack, l_thresh, u_thresh)
        if neighborhood_size > 0:
            if self.domain_range is None:
                self.domain = [(self.x0[i] - neighborhood_size, self.x0[i] + neighborhood_size) for i in range(self.x0.shape[0])]
            else:
                self.domain = [(max(self.x0[i] - neighborhood_size, self.domain_range[0]), min(self.x0[i] + neighborhood_size, self.domain_range[1])) for i in range(self.x0.shape[0])]

        if dc_argument is not None:
            if len(dc_argument.shape) == 0:
                # In case dc_argument is float (extreme eigenvalue) convert to matrix to get unified code for ADCD-E and ADCD-X
                dc_argument = dc_argument * numpy.eye(x0.shape[0])
            self.dc_argument = dc_argument

        start = timer()

        # This sync() is called as a result of full sync and therefore the coordinator sent update for the constraints.
        # Compute everything that can be computed now to save computations every call to set_new_data_point().
        g_func, h_func = self._get_dc(x0)
        if dc_type == DcType.Convex:
            self.g_func = g_func
            self.h_func = h_func
            self.g_func_grad_at_x0 = self.func_to_monitor_grad(x0).copy()  # The tangent line of g at the point x0
            self.h_func_grad_at_x0 = numpy.zeros(self.d)  # The tangent line of h at the point x0
        else:
            # The dc type is DcType.Concave.
            # To get unified code for convex diff and concave diff constraints, we convert concave diff to be convex diff.
            self.g_func = lambda x: -h_func(x)
            self.h_func = lambda x: -g_func(x)
            self.g_func_grad_at_x0 = numpy.zeros(self.d)  # The tangent line of g at the point x0
            self.h_func_grad_at_x0 = -self.func_to_monitor_grad(x0).copy()  # The tangent line of h at the point x0
        self.g_func_at_x0 = self.g_func(self.x0).copy()
        self.h_func_at_x0 = self.h_func(self.x0).copy()

        end = timer()
        self.full_sync_accumulated_time += end - start
        self.full_sync_accumulated_time_square += (end - start)**2
        self.full_sync_counter += 1

    # Override
    def dump_stats(self, test_folder):
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
        super().dump_stats(test_folder)

    # Override
    def _handle_sync_message(self, payload):
        constraint_version, x0, slack, l_thresh, u_thresh, neighborhood_size, dc_type, dc_argument = parse_message_sync_automon(payload, self.x0.shape[0])
        self.constraint_version = constraint_version
        self.sync(x0, slack, l_thresh, u_thresh, neighborhood_size, dc_type, dc_argument)
