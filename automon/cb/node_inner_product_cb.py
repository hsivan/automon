import numpy as np
from automon.cb.node_common_cb import NodeCommonCB

# Implementation according to https://dl.acm.org/doi/pdf/10.1145/3226113


class NodeInnerProductCB(NodeCommonCB):
    
    def __init__(self, idx=0, x0_len=2, domain=None, func_to_monitor=None):
        # func_to_monitor must be func_inner_product; however we keep function implementations outside of auto_mon core.
        NodeCommonCB.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_to_monitor)

    def _func_h(self, X, threshold):
        if len(X.shape) < 2:
            x = X[:X.shape[0]//2]
            y = X[X.shape[0]//2:]
            res = np.linalg.norm(x + y)**2
        else:
            x = X[:, :X.shape[1]//2]
            y = X[:, X.shape[1]//2:]
            res = np.linalg.norm(x + y, axis=1)**2
        return res 
    
    def _func_g(self, X, threshold):
        if len(X.shape) < 2:
            x = X[:X.shape[0]//2]
            y = X[X.shape[0]//2:]
            res = np.linalg.norm(x - y)**2 + 4 * threshold
        else:
            x = X[:, :X.shape[1]//2]
            y = X[:, X.shape[1]//2:]
            res = np.linalg.norm(x - y, axis=1)**2 + 4 * threshold
        return res
    
    def _func_h_grad(self, X, threshold):
        x = X[:X.shape[0]//2]
        y = X[X.shape[0]//2:]
        dh_dx = 2 * (x + y)
        dh_dy = 2 * (x + y)
        h_grad = np.concatenate((dh_dx, dh_dy))
        return h_grad
    
    def _func_g_grad(self, X, threshold):
        x = X[:X.shape[0]//2]
        y = X[X.shape[0]//2:]
        dh_dx = 2 * (x - y)
        dh_dy = -2 * (x - y)
        g_grad = np.concatenate((dh_dx, dh_dy))
        return g_grad
