import numpy as np
from automon.cb.cb_common_node import CbCommonNode

# Implementation according to https://dl.acm.org/doi/pdf/10.1145/3226113


class CbCosineSimilarityNode(CbCommonNode):
    
    def __init__(self, idx, func_to_monitor=None, d=2, domain=None):
        # func_to_monitor must be func_cosine_similarity; however we keep function implementations outside of automon core.
        CbCommonNode.__init__(self, idx, d=d, domain=domain, func_to_monitor=func_to_monitor)

    def _func_convexing_part(self, X, threshold):
        if len(X.shape) < 2:
            convexing_part = 2 * threshold * np.linalg.norm(X)**2
        else:
            convexing_part = 2 * threshold * np.linalg.norm(X, axis=1)**2
        if threshold < 0:
            convexing_part *= -1
        return convexing_part
    
    def _func_convexing_part_grad(self, X, threshold):
        x = X[:X.shape[0]//2]
        y = X[X.shape[0]//2:]
        dh_dx = 4 * threshold * x
        dh_dy = 4 * threshold * y
        convexing_part_grad = np.concatenate((dh_dx, dh_dy))
        if threshold < 0:
            convexing_part_grad *= -1
        return convexing_part_grad
        
    def _func_h(self, X, threshold):
        if len(X.shape) < 2:
            x = X[:X.shape[0]//2]
            y = X[X.shape[0]//2:]
            res = np.linalg.norm(x + y)**2
        else:
            x = X[:, :X.shape[1]//2]
            y = X[:, X.shape[1]//2:]
            res = np.linalg.norm(x + y, axis=1)**2
        res += self._func_convexing_part(X, threshold)
        return res 
    
    def _func_g(self, X, threshold):
        if len(X.shape) < 2:
            x = X[:X.shape[0]//2]
            y = X[X.shape[0]//2:]
            res = np.linalg.norm(x - y)**2 + 4 * threshold * np.linalg.norm(x) * np.linalg.norm(y)
        else:
            x = X[:, :X.shape[1]//2]
            y = X[:, X.shape[1]//2:]
            res = np.linalg.norm(x - y, axis=1)**2 + 4 * threshold * np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        res += self._func_convexing_part(X, threshold)
        return res
    
    def _func_h_grad(self, X, threshold):
        x = X[:X.shape[0]//2]
        y = X[X.shape[0]//2:]
        dh_dx = 2 * (x + y)
        dh_dy = 2 * (x + y)
        h_grad = np.concatenate((dh_dx, dh_dy)) + self._func_convexing_part_grad(X, threshold)
        return h_grad
    
    def _func_g_grad(self, X, threshold):
        x = X[:X.shape[0]//2]
        y = X[X.shape[0]//2:]
        
        if np.all(x == 0.0):
            dh_dx = 2 * (x - y)
        else:
            dh_dx = 2 * (x - y) + 4 * threshold * np.linalg.norm(y) * (x / np.linalg.norm(x))
        
        if np.all(y == 0.0):
            dh_dy = -2 * (x - y)
        else:
            dh_dy = -2 * (x - y) + 4 * threshold * np.linalg.norm(x) * (y / np.linalg.norm(y))
        
        g_grad = np.concatenate((dh_dx, dh_dy)) + self._func_convexing_part_grad(X, threshold)
        return g_grad
