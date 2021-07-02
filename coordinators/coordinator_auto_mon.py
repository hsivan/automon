import enum
from scipy.optimize import minimize
from timeit import default_timer as timer
import logging
from coordinators.coordinator_common import CoordinatorCommon, SlackType, SyncType
import numpy
import os

# Can set this environment variable in test to force use AutoGrad, even if Jax exists: os.environ['AUTO_GRAD_TOOL'] = 'AutoGrad'
AUTO_GRAD_TOOL = os.getenv('AUTO_GRAD_TOOL')
# Try import Jax if AUTO_GRAD_TOOL is None (automatic decision, try Jax if exists) or AUTO_GRAD_TOOL is Jax.
# If failed to import Jax or AUTO_GRAD_TOOL is AutoGrad use AutoGrad
if (AUTO_GRAD_TOOL is None) or (AUTO_GRAD_TOOL == "Jax"):
    try:
        import jax.numpy as np
        from jax import jit, jacfwd, jacrev
        from jax import grad as jax_grad
        from jax.config import config
        import jax
        config.update("jax_enable_x64", True)
        print("Import Jax as np")
        AUTO_GRAD_TOOL = "Jax"

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
        print("Can't import Jax module")
        AUTO_GRAD_TOOL = "AutoGrad"
if AUTO_GRAD_TOOL == "AutoGrad":
    import autograd.numpy as np
    from autograd import hessian, grad

    def mineigenval(x, hess):
        eigs = np.linalg.eigvalsh(hess(x))  # The Hessian is a real symmetric matrix
        return np.min(eigs)

    def maxeigenval(x, hess):
        eigs = np.linalg.eigvalsh(hess(x))  # The Hessian is a real symmetric matrix
        return np.max(eigs)

    print("Import Autograd as np")


class DomainType(enum.Enum):
    # The domain is simply the given domain.
    Absolute = 0
    # The domain is relative to the reference point x0: a neighborhood around x0.
    Relative = 1


class DcType(enum.Enum):
    Convex = 0
    Concave = 1


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
        num_iterations = 3
        min_eigenvalue = numpy.inf
        max_eigenvalue = -numpy.inf
        if domain is None:
            rand_start_points = numpy.random.uniform(x0-3, x0+3, (num_iterations, x0.shape[0]))
        else:
            rand_start_points = numpy.random.uniform([min_entry for min_entry, _ in domain], [max_entry for _, max_entry in domain], (num_iterations, x0.shape[0]))
        for start_point in rand_start_points:
            start_point = np.array([start_point], dtype=np.float32)
            sol_min = minimize(self._min_eigenvalue, start_point, args=self.func_to_monitor_hessian, bounds=domain, options={"maxiter": 5}, method=self.optimization_method)
            sol_max = minimize(self._max_eigenvalue, start_point, args=self.func_to_monitor_hessian, bounds=domain, options={"maxiter": 5}, method=self.optimization_method)
            # logging.info("Iteration " + str(iteration) + ": Minimal eigenvalue of hessian in the domain: " + str(sol_min.fun) + " at " + str(sol_min.x))
            # logging.info("Iteration " + str(iteration) + ": Maximal eigenvalue of hessian in the domain: " + str(-1.0 * sol_max.fun) + " at " + str(sol_max.x))
            min_eigenvalue = numpy.minimum(min_eigenvalue, sol_min.fun)
            max_eigenvalue = numpy.maximum(max_eigenvalue, -1.0 * sol_max.fun)

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

    def _adcd_e_gradients(self, signed_H, x0):
        g_func_grad = lambda x: self.func_to_monitor_grad(x) - (x - x0) @ signed_H
        h_func_grad = lambda x: -(x - x0) @ signed_H
        return g_func_grad, h_func_grad
    
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

        g_func_grad, h_func_grad = self._adcd_e_gradients(self.signed_H, x0)
        return self.dc_type, self.signed_H, g_func_grad, h_func_grad
    
    # ADCD-X.

    def _adcd_x_gradients(self, extreme_lambda, x0):
        g_func_grad = lambda x: self.func_to_monitor_grad(x) - extreme_lambda * (x - x0).T
        h_func_grad = lambda x: -extreme_lambda * (x - x0).T
        return g_func_grad, h_func_grad
    
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
        g_func_grad, h_func_grad = self._adcd_x_gradients(extreme_lambda, x0)
                
        return dc_type, extreme_lambda, g_func_grad, h_func_grad


class CoordinatorAutoMon(CoordinatorCommon):
    
    def __init__(self, verifier, func_to_monitor, x0_len, num_nodes, error_bound=2,
                 slack_type=SlackType.Drift, sync_type=SyncType.Eager, lazy_sync_max_S=0.5, domain=None,
                 domain_type=DomainType.Absolute, neighborhood_size=1.0):
        CoordinatorCommon.__init__(self, verifier, func_to_monitor, x0_len, num_nodes, error_bound, slack_type, sync_type, lazy_sync_max_S)
        logging.info("CoordinatorAutoMon initialization: domain " + str(domain) + ", domain_type " + str(domain_type) + ", AUTO_GRAD_TOOL " + AUTO_GRAD_TOOL + ", neighborhood_size " + str(neighborhood_size))
        self.coordinator_name = "AutoMon"
        self.b_violation_strict = False
        self.x0_len = x0_len

        self.domain = domain
        self.domain_type = domain_type
        # If domain_type is relative (fixed or adaptive), then neighborhood is updated according to x0. Otherwise it remains domain.
        self.neighborhood = [domain] * x0_len if domain is not None else None
        # Relevant only if domain_type is Relative and Hessian is not constant
        self.neighborhood_size = neighborhood_size
        # This flag is set to True only during neighborhood size tuning procedure
        self.b_tune_neighborhood_mode = False
        # This flag is set to False only during neighborhood size tuning procedure or fixed neighborhood size experiments
        self.b_fix_neighborhood_dynamically = True
        # Relevant only if self.b_fix_neighborhood_dynamically is True:
        # If more than this threshold consecutive neighborhood violations (without safe-zone violations), need to
        # increase neighborhood_size by 2.
        self.neighborhood_violation_counter_threshold = num_nodes * 5
        self.update_neighborhood_counter = 0

        self.adcd_helper = AdcdHelper(func_to_monitor)
        # Check if the Hessian is constant. If the Hessian const there is no need to find the min and max eigenvalues for every x0 update.
        # Also, for constant hessian the neighborhood remains the entire domain during the test.
        self.b_hessian_const = self._is_hessian_const(x0_len)
    
    def _update_neighborhood(self):
        if self.domain_type == DomainType.Absolute:
            return

        # Domain type is DomainType.Relative. Should update the neighborhood around x0.
        if self.b_fix_neighborhood_dynamically and self.consecutive_neighborhood_violations_counter > self.neighborhood_violation_counter_threshold:
            self.neighborhood_size *= 2
            logging.info("Iteration " + str(self.iteration) + ": Update neighborhood size to " + str(self.neighborhood_size))
        if self.domain is None:
            self.neighborhood = [(self.x0[i] - self.neighborhood_size, self.x0[i] + self.neighborhood_size) for i in range(self.x0_len)]
        else:
            self.neighborhood = [(max(self.x0[i] - self.neighborhood_size, self.domain[0]), min(self.x0[i] + self.neighborhood_size, self.domain[1])) for i in range(self.x0_len)]

        logging.info("Iteration " + str(self.iteration) + ": Update neighborhood to " + str(self.neighborhood))

        self.update_neighborhood_counter += 1
        if self.b_tune_neighborhood_mode and self.update_neighborhood_counter == 2:
            # This is the first sync after windows of all nodes are full. Should ignore all violations up until now when
            # in neighborhood size tuning mode.
            self.statistics.total_violations_msg_counter = 0
            self.statistics.violation_origin_outside_safe_zone = 0
            self.statistics.violation_origin_outside_domain = 0

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
        self.verifier.sync(self.x0, numpy.zeros_like(self.x0), self.l_thresh, self.u_thresh, self.neighborhood, self.dc_type, self.dc_argument, self.g_func_grad_at_x0, self.h_func_grad_at_x0)
    
    # Override - difference in sync
    def _sync_node(self, node_idx, sync_type="full"):
        node = self.nodes[node_idx]
        if sync_type == "full":
            # Only at full sync we update the thresholds, neighborhood, g_func, h_func, g_func_grad, and h_func_grad
            node.sync(self.x0, self.nodes_slack[node_idx], self.l_thresh, self.u_thresh, self.neighborhood, self.dc_type, self.dc_argument, self.g_func_grad_at_x0, self.h_func_grad_at_x0)
        else:
            node.sync(self.x0, self.nodes_slack[node_idx])
    
    # Override
    def _update_l_u_threshold(self):
        super()._update_l_u_threshold()
        if not self.b_hessian_const:
            # Using neighborhood helps mitigating the constraints if the extreme eigenvalues are "less extreme" in smaller neighborhood.
            # However, for fixed Hessian (and therefore fixed eigenvalues) the neighborhood should be the entire domain size.
            self._update_neighborhood()
            # Since the Hessian is non const, we use ADCD-X
            dc_type, dc_argument, g_func_grad, h_func_grad = self.adcd_helper.adcd_x(self.x0, self.neighborhood, self.iteration)
        else:
            # Since the Hessian is const, we use ADCD-E
            dc_type, dc_argument, g_func_grad, h_func_grad = self.adcd_helper.adcd_e(self.x0)

        self.dc_type = dc_type
        self.dc_argument = dc_argument
        self.g_func_grad_at_x0 = g_func_grad(self.x0).copy()
        self.h_func_grad_at_x0 = h_func_grad(self.x0).copy()
    
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

        if test_folder is not None:
            file_prefix = test_folder + "/" + self.coordinator_name
            with open(file_prefix + "_optimization_times.csv", 'wb') as f:
                numpy.savetxt(f, optimization_history_times)

        return res
