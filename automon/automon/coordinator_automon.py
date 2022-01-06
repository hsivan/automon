from scipy.optimize import minimize
from timeit import default_timer as timer
import logging
from automon.coordinator_common import CoordinatorCommon, SlackType, SyncType
import numpy
import os
from automon.messages_common import prepare_message_lazy_sync, ViolationOrigin
from automon.automon.messages_automon import DcType, prepare_message_sync_automon

logging = logging.getLogger(__name__)

# Can set this environment variable in test to force use AutoGrad, even if Jax exists: os.environ['AUTO_DIFFERENTIATION_TOOL'] = 'AutoGrad'
AUTO_DIFFERENTIATION_TOOL = os.getenv('AUTO_DIFFERENTIATION_TOOL')
# Try import Jax if AUTO_DIFFERENTIATION_TOOL is None (automatic decision, try Jax first) or AUTO_DIFFERENTIATION_TOOL is Jax.
# If failed to import Jax or AUTO_DIFFERENTIATION_TOOL is AutoGrad use AutoGrad
if (AUTO_DIFFERENTIATION_TOOL is None) or (AUTO_DIFFERENTIATION_TOOL == "Jax"):
    try:
        import jax.numpy as np
        from jax import jit, jacfwd, jacrev
        from jax import grad as jax_grad
        from jax.config import config
        import jax
        config.update("jax_enable_x64", True)
        config.update("jax_platform_name", 'cpu')
        AUTO_DIFFERENTIATION_TOOL = "Jax"

        def hessian(fun):
            return jit(jacfwd(jacrev(fun)))

        def grad(fun):
            return jit(jax_grad(fun))

        @jax.partial(jit, static_argnums=1)
        def mineigenval(x, hess):
            eigs = np.linalg.eigvalsh(hess(x))  # The Hessian is a real symmetric matrix
            return np.min(eigs)

        @jax.partial(jit, static_argnums=1)
        def maxeigenval(x, hess):
            eigs = np.linalg.eigvalsh(hess(x))  # The Hessian is a real symmetric matrix
            return np.max(eigs)

    except Exception as e:
        AUTO_DIFFERENTIATION_TOOL = "AutoGrad"
if AUTO_DIFFERENTIATION_TOOL == "AutoGrad":
    import autograd.numpy as np
    from autograd import hessian, grad

    def mineigenval(x, hess):
        eigs = np.linalg.eigvalsh(hess(x))  # The Hessian is a real symmetric matrix
        return np.min(eigs)

    def maxeigenval(x, hess):
        eigs = np.linalg.eigvalsh(hess(x))  # The Hessian is a real symmetric matrix
        return np.max(eigs)


# Use HIPS AutoGrad or Jax for minimization/maximization of eigenvalues
class ExtremeEigenvalueHelper:
    
    def __init__(self, func_to_monitor):
        self.func_to_monitor_hessian = hessian(func_to_monitor)
        self.optimization_history_times = []

        # There is a bug in SLSQP in scipy < 1.6.0: https://github.com/scipy/scipy/pull/13009 , therefore using L-BFGS-B
        self.optimization_method = "L-BFGS-B"
        logging.info("Optimization method is " + self.optimization_method)
    
    # Minimize this function over x in a specific neighborhood around X0
    def _min_eigenvalue(self, x, args):
        hess = args
        min_eigenvalue = mineigenval(x, hess)
        return min_eigenvalue

    # Maximize this function over x in a specific neighborhood around X0
    def _max_eigenvalue(self, x, args):
        hess = args
        max_eigenvalue = maxeigenval(x, hess)
        return -1.0 * max_eigenvalue
    
    def func_extreme_eigenvalues(self, x0, domain, iteration):
        start = timer()
        sol_min = minimize(self._min_eigenvalue, x0, args=self.func_to_monitor_hessian, bounds=domain, options={"maxiter": 5}, method=self.optimization_method)
        sol_max = minimize(self._max_eigenvalue, x0, args=self.func_to_monitor_hessian, bounds=domain, options={"maxiter": 5}, method=self.optimization_method)
        logging.debug("Iteration " + str(iteration) + ": Minimal eigenvalue of hessian in the domain: " + str(sol_min.fun) + " at " + str(sol_min.x))
        logging.debug("Iteration " + str(iteration) + ": Maximal eigenvalue of hessian in the domain: " + str(-1.0 * sol_max.fun) + " at " + str(sol_max.x))
        min_eigenvalue = sol_min.fun
        max_eigenvalue = -1.0 * sol_max.fun

        if numpy.isnan(min_eigenvalue):
            min_eigenvalue = -numpy.inf
        if numpy.isnan(max_eigenvalue):
            max_eigenvalue = numpy.inf

        logging.info("Iteration " + str(iteration) + ": Minimal eigenvalue of hessian in the domain: " + str(min_eigenvalue))
        logging.info("Iteration " + str(iteration) + ": Maximal eigenvalue of hessian in the domain: " + str(max_eigenvalue))
        end = timer()
        optimization_time = end - start
        logging.info("Optimization time: " + str(optimization_time))
        self.optimization_history_times.append(end - start)
        assert (min_eigenvalue <= max_eigenvalue)
        return min_eigenvalue, max_eigenvalue
    
    def eigenvalues_eigenvectors(self, x):
        eigenvalues, eigenvectors = numpy.linalg.eigh(self.func_to_monitor_hessian(x))  # The Hessian is a real symmetric matrix
        return eigenvalues, eigenvectors


# Find the best difference representation: if the Hessian is constant use ADCD-E, otherwise use ADCD-X.
class AdcdHelper:
    
    def __init__(self, func_to_monitor):
        self.func_to_monitor = func_to_monitor
        self.func_to_monitor_grad = grad(func_to_monitor)
        self.extreme_eigenvalue_helper = ExtremeEigenvalueHelper(func_to_monitor)
        self.signed_H = None
        self.dc_type = None

    # ADCD-E.

    def _get_signed_H(self, eigenvalues, eigenvectors, b_minus):
        # Returns H_plus if b_minus is True, otherwise returns H_plus
        signed_eigenvalues = eigenvalues.copy()
        if b_minus:
            signed_eigenvalues[eigenvalues > 0] = 0
        else:
            signed_eigenvalues[eigenvalues < 0] = 0
        signed_diag = numpy.diag(signed_eigenvalues)
        signed_H = eigenvectors @ signed_diag @ eigenvectors.T
        return signed_H
    
    def adcd_e(self, x0):
        if self.signed_H is None:
            # Done only once at the first call to this function.
            # Hessian is similar for every x0 as it constant, and therefore we only need to evaluate H_minus or H_plus once.
            hess = hessian(self.func_to_monitor)
            eigenvalues, eigenvectors = numpy.linalg.eigh(hess(x0))  # The Hessian is a real symmetric matrix
            lambda_min, lambda_max = eigenvalues[0], eigenvalues[-1]
            assert(lambda_min <= lambda_max)

            if numpy.abs(lambda_min) <= lambda_max:
                logging.info("Use g,h convex diff, ADCD-E with H_minus.")
                dc_type = DcType.Convex
                signed_H = self._get_signed_H(eigenvalues, eigenvectors, True)
            else:
                logging.info("Use g,h concave diff, ADCD-E with H_plus.")
                dc_type = DcType.Concave
                signed_H = self._get_signed_H(eigenvalues, eigenvectors, False)

            self.signed_H = signed_H
            self.dc_type = dc_type

        return self.dc_type, self.signed_H
    
    # ADCD-X.
    
    def adcd_x(self, x0, domain, iteration):
        min_eigenvalue, max_eigenvalue = self.extreme_eigenvalue_helper.func_extreme_eigenvalues(x0, domain, iteration)
        lambda_min_minus = numpy.minimum(min_eigenvalue, 0)
        lambda_max_plus = numpy.maximum(max_eigenvalue, 0)

        eigenvalues, eigenvectors = self.extreme_eigenvalue_helper.eigenvalues_eigenvectors(x0)
        min_eigenvalue_at_x0, max_eigenvalue_at_x0 = eigenvalues[0], eigenvalues[-1]
        # Decide whether to use convex diff or concave diff according to the DH Heuristic: lambda_min(H_g(x0))+lambda_min(H_h) <= abs(lambda_max(H_h_hat)+lambda_max(H_g_hat(x0))).
        lambda_min_g, lambda_min_h = min_eigenvalue_at_x0 + numpy.abs(lambda_min_minus), numpy.abs(lambda_min_minus)  # Both values should be >= 0
        lambda_max_h_hat, lambda_max_g_hat = -1.0 * lambda_max_plus, max_eigenvalue_at_x0 - 1.0 * lambda_max_plus  # Both values should be <= 0

        if numpy.abs(lambda_min_g + lambda_min_h) <= numpy.abs(lambda_max_h_hat + lambda_max_g_hat):
            logging.info("Use g,h convex diff, ADCD-X with lambda_min_minus " + str(lambda_min_minus))
            dc_type = DcType.Convex
            extreme_lambda = lambda_min_minus
        else:
            logging.info("Use g,h concave diff, ADCD-X with lambda_max_plus " + str(lambda_max_plus))
            dc_type = DcType.Concave
            extreme_lambda = lambda_max_plus

        return dc_type, extreme_lambda


class CoordinatorAutoMon(CoordinatorCommon):
    
    def __init__(self, verifier, num_nodes, error_bound=2, slack_type=SlackType.Drift, sync_type=SyncType.LazyLRU,
                 lazy_sync_max_S=0.5, neighborhood_size=None):
        CoordinatorCommon.__init__(self, verifier, num_nodes, error_bound, slack_type, sync_type, lazy_sync_max_S,
                                   b_violation_strict=False, coordinator_name="AutoMon")
        logging.info("CoordinatorAutoMon initialization: domain " + str(verifier.domain_range) + ", AUTO_DIFFERENTIATION_TOOL " + AUTO_DIFFERENTIATION_TOOL + ", neighborhood_size " + str(neighborhood_size))
        self.adcd_helper = AdcdHelper(verifier.func_to_monitor)

        self.domain = verifier.domain_range  # If None, the domain is the entire R^d
        # If neighborhood_size is not None, then neighborhood is updated according to x0. Otherwise it remains the entire domain.
        self.neighborhood = [verifier.domain_range] * self.x0_len if verifier.domain_range is not None else None
        # Check if the Hessian is constant. If the Hessian const there is no need to find the min and max eigenvalues for every x0 update.
        # Also, for constant hessian the neighborhood remains the entire domain during the test.
        self.b_hessian_const = self._is_hessian_const(self.x0_len)
        if self.b_hessian_const or neighborhood_size is None:
            self.initial_neighborhood_size = -1
        else:
            # Relevant only if neighborhood_size is given and Hessian is not constant
            self.initial_neighborhood_size = neighborhood_size
        # Relevant only if self.b_fix_neighborhood_dynamically is True:
        # If more than this threshold consecutive neighborhood violations (without safe-zone violations), need to
        # increase neighborhood_size by 2.
        self.neighborhood_violation_counter_threshold = num_nodes * 5

        self._init()

    def _init(self):
        super()._init()
        self.consecutive_neighborhood_violations_counter = 0
        # If initial_neighborhood_size is not -1 (neighborhood_size is given), then neighborhood is updated according to x0. Otherwise it remains the entire domain.
        self.neighborhood = [self.domain] * self.x0_len if self.domain is not None else None
        # Relevant only if neighborhood_size is given and Hessian is not constant
        self.neighborhood_size = self.initial_neighborhood_size
        # This flag is set to True only during neighborhood size tuning procedure
        self.b_tune_neighborhood_mode = False
        # This flag is set to False only during fixed neighborhood size experiments. It is ignored during neighborhood size tuning procedure.
        self.b_fix_neighborhood_dynamically = True

    def _is_neighborhood_size_update_required(self):
        # Return if neighborhood size update is required.
        # make sure not in tuning mode - should not change neighborhood size dynamically during tuning.
        return (not self.b_tune_neighborhood_mode) and self.b_fix_neighborhood_dynamically and (self.consecutive_neighborhood_violations_counter > self.neighborhood_violation_counter_threshold)
    
    def _update_neighborhood(self):
        # Using neighborhood helps mitigating the constraints if the extreme eigenvalues are "less extreme" in smaller neighborhood.
        # However, for fixed Hessian (and therefore fixed eigenvalues) the neighborhood should be the entire domain size.
        # Moreover, if no neighborhood_size was given in initialization, the neighborhood should be the entire domain.
        if self.b_hessian_const or self.initial_neighborhood_size == -1:
            return

        # Update the neighborhood around x0.

        if self._is_neighborhood_size_update_required():
            self.neighborhood_size *= 2
            self.consecutive_neighborhood_violations_counter = 0
            logging.info("Iteration " + str(self.iteration) + ": Update neighborhood size to " + str(self.neighborhood_size))

        if self.domain is None:
            self.neighborhood = [(self.x0[i] - self.neighborhood_size, self.x0[i] + self.neighborhood_size) for i in range(self.x0_len)]
        else:
            self.neighborhood = [(max(self.x0[i] - self.neighborhood_size, self.domain[0]), min(self.x0[i] + self.neighborhood_size, self.domain[1])) for i in range(self.x0_len)]

        logging.info("Iteration " + str(self.iteration) + ": Update neighborhood to " + str(self.neighborhood))

    # We check if the Hessian is x-dependent or not.
    def _is_hessian_const(self, x_shape):
        b_constant_hessian = True

        if self.neighborhood is None:
            rand_start_points = numpy.random.uniform(-10, 10, (3, x_shape))
        else:
            rand_start_points = numpy.random.uniform([min_entry for min_entry, _ in self.neighborhood], [max_entry for _, max_entry in self.neighborhood], (3, x_shape))

        start_point = np.array(rand_start_points[0], dtype=np.float32)
        hess = hessian(self.func_to_monitor)
        hess_at_first_point = hess(start_point)

        for start_point in rand_start_points[1:]:
            start_point = np.array(start_point, dtype=np.float32)
            hess_at_other_point = hess(start_point)
            # Check if the Hessians are equal
            if not np.array_equal(hess_at_first_point, hess_at_other_point):
                b_constant_hessian = False

        if b_constant_hessian:
            # The Hessian is const. We use ADCD-E only once at initialization.
            logging.info("Found indication that the Hessian is constant. Use ADCD-E technique.")
            b_constant_hessian = True
        return b_constant_hessian
    
    # Override - difference in sync
    def _sync_verifier(self):
        # Since verifier.x equals new_x0, no slack is ever needed.
        self.verifier.sync(self.x0, numpy.zeros_like(self.x0), self.l_thresh, self.u_thresh, self.neighborhood_size, self.dc_type, self.dc_argument)
    
    # Override - difference in sync
    def _sync_node(self, node_idx, sync_type="full"):
        self.nodes_constraint_version[node_idx] = self.iteration
        if sync_type == "full":
            # Only at full sync we update the local constraints
            if self.b_hessian_const and self.iteration > 0:
                # No need to send dc_argument more than once if using ADCD-E (and on iteration 0 all the nodes are synced).
                # dc_argument doesn't change in ADCD-E, and moreover, it is a matrix and not a scalar, and therefore should be sent only once to reduce the message size.
                message_out = prepare_message_sync_automon(node_idx, self.nodes_constraint_version[node_idx], self.x0, self.nodes_slack[node_idx], self.l_thresh, self.u_thresh,
                                                           self.neighborhood_size, self.dc_type, None)
            else:
                message_out = prepare_message_sync_automon(node_idx, self.nodes_constraint_version[node_idx], self.x0, self.nodes_slack[node_idx], self.l_thresh, self.u_thresh,
                                                           self.neighborhood_size, self.dc_type, self.dc_argument)
        else:
            message_out = prepare_message_lazy_sync(node_idx, self.nodes_constraint_version[node_idx], self.nodes_slack[node_idx])
        return message_out

    # Override
    def _is_eager_sync_required(self):
        return self._is_neighborhood_size_update_required()

    # Override
    def _notify_violation(self, node_idx, violation_origin):
        if violation_origin == ViolationOrigin.Domain:
            self.consecutive_neighborhood_violations_counter += 1
        else:
            self.consecutive_neighborhood_violations_counter = 0
        super()._notify_violation(node_idx, violation_origin)
    
    # Override
    def _update_l_u_threshold(self):
        super()._update_l_u_threshold()
        self._update_neighborhood()
        if not self.b_hessian_const:  # for non-const Hessian use ADCD-X
            dc_type, dc_argument = self.adcd_helper.adcd_x(self.x0, self.neighborhood, self.iteration)
        else:  # for const Hessian use ADCD-E
            dc_type, dc_argument = self.adcd_helper.adcd_e(self.x0)

        self.dc_type = dc_type
        self.dc_argument = dc_argument
    
    # Override - add extra statistics regarding running time
    def dump_stats(self, test_folder):
        res = super().dump_stats(test_folder)
        
        optimization_history_times = self.adcd_helper.extreme_eigenvalue_helper.optimization_history_times
        if self.b_hessian_const:
            assert(len(optimization_history_times) == 0)
        else:
            logging.info("Optimization history len " + str(len(optimization_history_times)))
            if len(optimization_history_times) - 1 > 0:
                # Ignore first optimization as it contains jit compilation
                logging.info("Avg optimization time " + str(numpy.mean(optimization_history_times[1:])))
                logging.info("Std optimization time " + str(numpy.std(optimization_history_times[1:])))
                logging.info("Max optimization time " + str(numpy.max(optimization_history_times[1:])))

        if test_folder is not None:
            file_prefix = test_folder + "/" + self.coordinator_name
            with open(file_prefix + "_optimization_times.csv", 'wb') as f:
                numpy.savetxt(f, optimization_history_times)

        return res
