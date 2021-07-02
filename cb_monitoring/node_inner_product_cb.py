import numpy as np
from cb_monitoring.node_common_cb import NodeCommonCB


class NodeInnerProductCB(NodeCommonCB):
    
    def __init__(self, idx=0, local_vec_len=2):
        NodeCommonCB.__init__(self, idx, initial_x0=np.zeros(local_vec_len))

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
