import numpy as np
from automon.gm.node_common_gm import NodeCommonGM
import scipy as sp
from scipy.optimize import NonlinearConstraint

# Implementation according to https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6877240


def func_q_x0_distance(q, x0):
    return np.linalg.norm(q - x0, 2) ** 2


def func_q_on_parabola(q):
    return q[1] - q[0]**2


class NodeVarianceGM(NodeCommonGM):
    
    def __init__(self, idx=0, x0_len=2, domain=None, func_to_monitor=None):
        # func_to_monitor must be func_variance; however we keep function implementations outside of automon core.
        assert (x0_len == 2)  # The local vector is the first and second momentum
        NodeCommonGM.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_to_monitor)

    def _calc_parabola(self, thresh, x):
        # Calculates y = x**2 + thresh
        y = x**2 + thresh
        return y
    
    def _calc_q_x_for_negative_discriminant(self, R, Q):
        theta = np.arccos(R / np.sqrt(-Q**3))
        # There are 3 real roots
        roots = 2 * np.sqrt(-Q) * np.cos(theta / 3 + np.array([0, 2, 4]) * np.pi / 3)
        # Choose the root that is closest to self.x0[0], and set q_x to it
        closest_root_index = np.argmin(np.abs(self.x0[0] - roots))
        q_x = roots[closest_root_index]
        self.roots = roots
        return q_x
    
    def _calc_q_x_for_positive_discriminant(self, R, Q):
        discriminant_sqrt = np.sqrt(R**2 + Q**3)
        q_x = np.cbrt(R + discriminant_sqrt) + np.cbrt(R - discriminant_sqrt)
        self.roots = np.array([q_x])
        return q_x        
        
    def _calc_q_numerically(self):
        constraint = NonlinearConstraint(func_q_on_parabola, self.u_thresh, self.u_thresh)
        q = sp.optimize.minimize(func_q_x0_distance, self.x0, args=self.x0, constraints=(constraint))
        return q
    
    def _calc_q(self):
        # Taken from: https://proofwiki.org/wiki/Cardano%27s_Formula
        Q = ((2 * self.u_thresh + 1 - 2 * self.x0[1]) / 2) / 3
        R = -(-0.5 * self.x0[0]) / 2
        discriminant = R**2 + Q**3

        if discriminant < 0:
            q_x = self._calc_q_x_for_negative_discriminant(R, Q)
            # Make sure that when the discriminant is negative then self.x0[1]
            # is above the lowest point of the parabola.
            parabola_min_y = self.u_thresh
            assert(self.x0[1] > parabola_min_y)
        else:
            q_x = self._calc_q_x_for_positive_discriminant(R, Q)
            # Note that the discriminant can be positive and still self.x0[1]
            # is above the lowest point of the parabola. However, it is always
            # negative when self.x0[1] is below the lowest point of the parabola.
        
        q_y = self._calc_parabola(self.u_thresh, q_x)
        q = np.array([q_x, q_y])
        
        q_numeric = self._calc_q_numerically()
        assert(np.all(q - q_numeric.x < 1e-4))
        
        return q
        
    def _below_safe_zone_upper_bound(self, x):
        x0_minus_q = self.x0 - self.q
        x_minus_q = x - self.q
        b_inside_safe_zone = True if x0_minus_q @ x_minus_q >= 0 else False
        return b_inside_safe_zone
    
    def _above_safe_zone_lower_bound(self, x):
        f = self.func_to_monitor(x)
        b_inside_safe_zone = True if f >= self.l_thresh else False
        return b_inside_safe_zone
