import numpy as np
from automon.gm.gm_common_node import GmCommonNode
import cvxpy as cp

# Implementation according to https://dl.acm.org/doi/pdf/10.1145/3097983.3098092


class GmEntropyNode(GmCommonNode):
    
    def __init__(self, idx=0, x0_len=1, domain=None, func_to_monitor=None):
        # func_to_monitor must be func_entropy; however we keep function implementations outside of automon core.
        GmCommonNode.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_to_monitor)
        self.k = x0_len
        # point_to_check is a probability vector, however, it is possible to
        # get negative or above 1 values when using drift slack.
        # In this case we should report violation or use tilda_f as in CIDER paper.
        # Consider adding implementation of tilda_f to extend the domain of the function (see https://dl.acm.org/doi/pdf/10.1145/3097983.3098092, 4.3).

    def _calc_q(self):
        # Use CVXPY to solve the following optimization problem:
        # q = argmin_x ||x - x0||^2 s.t. self.u_thresh - func_entropy(x) <= 0
        
        x0 = self.x0
        q = cp.Variable(self.k)
            
        problem = cp.norm(q - x0, 2)**2
        constraints = [self.u_thresh - cp.sum(cp.entr(q)) <= 0, cp.sum(q) == 1]
        
        objective = cp.Minimize(problem)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        q = q.value
        
        if q is None:
            # Entropy gets the maximum for uniform P
            vector_for_max_entropy = np.ones(self.k) / self.k
            max_entropy = self.func_to_monitor(vector_for_max_entropy)
            assert(self.u_thresh >= max_entropy)
            # Set q to be the point that gives the maximum entropy
            q = vector_for_max_entropy
        return q
    
    def _below_safe_zone_upper_bound(self, v):
        # There is a special case, in which u_thresh is above the maximum point
        # of the entropy f(). In this case we set q to be the vector that gives this maximum
        # value of f(), however, there is no tangent line (the tangent hyperplane to f() at q is
        # parallel to the xy plane, touches f() at the top).
        # In that special case, every point is under the safe zone upper bound.
        # Here we check for this case.
        vector_for_max_entropy = np.ones(self.k) / self.k
        max_entropy = self.func_to_monitor(vector_for_max_entropy)
        if self.u_thresh >= max_entropy:
            b_inside_safe_zone = True
            return b_inside_safe_zone
        
        # This is the regular case, where u_thresh is bellow the maximum of f().
        x0_minus_q = self.x0 - self.q
        p_minus_q = v - self.q
        b_inside_safe_zone = True if x0_minus_q @ p_minus_q >= 0 else False
        return b_inside_safe_zone
    
    def _above_safe_zone_lower_bound(self, v):
        f = self.func_to_monitor(v)
        b_inside_safe_zone = True if f >= self.l_thresh else False
        return b_inside_safe_zone
