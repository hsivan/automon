import os
import numpy
# Can set this environment variable in test to force use AutoGrad, even if Jax exists: os.environ['AUTO_DIFFERENTIATION_TOOL'] = 'AutoGrad'/'Jax'
AUTO_DIFFERENTIATION_TOOL = os.getenv('AUTO_DIFFERENTIATION_TOOL')
# Try import Jax if AUTO_DIFFERENTIATION_TOOL is None (automatic decision, try Jax first) or AUTO_DIFFERENTIATION_TOOL is Jax.
# If failed to import Jax or AUTO_DIFFERENTIATION_TOOL is AutoGrad use AutoGrad
if (AUTO_DIFFERENTIATION_TOOL is None) or (AUTO_DIFFERENTIATION_TOOL == "Jax"):
    try:
        import jax
        from jax import jit
        import jax.numpy as np
        from jax.config import config
        config.update("jax_enable_x64", True)
        config.update("jax_platform_name", 'cpu')
        #print("Use Jax implementation for functions to monitor")
        AUTO_DIFFERENTIATION_TOOL = "Jax"
        key = jax.random.PRNGKey(0)
    except Exception as e:
        AUTO_DIFFERENTIATION_TOOL = "AutoGrad"
if AUTO_DIFFERENTIATION_TOOL == "AutoGrad":
    import autograd.numpy as np
    #print("Use AutoGrad implementation for functions to monitor")


def maybe_jit(func):
    return jit(func) if AUTO_DIFFERENTIATION_TOOL == "Jax" else func


############ Entropy ############
@maybe_jit
def func_entropy(p_vec):
    # Compute entropy function.
    # p_vec is a probability array of the number of observations for each value from 1 to k.

    if len(p_vec.shape) >= 2:  # For figures only, not used for grad or Hessian
        p = p_vec
        p[p == 0] = 1
        log_p = np.log(p)
        return -np.sum(p_vec * log_p, axis=1)

    p = np.where(p_vec > 0, p_vec, 1)
    entropy_p = np.where(p_vec > 0, -1.0 * p * np.log(p), 0)
    res = np.sum(entropy_p, axis=0)
    return res


############ DNN Exp: x*exp(-x^2 -y^2)   and   DNN intrusion detection ############

network_params = None
net_apply = None


def set_net_params(network_params_, net_apply_):
    global network_params
    global net_apply
    network_params = network_params_
    net_apply = net_apply_


# Only works with Jax
@maybe_jit
def func_mlp(X):
    global network_params
    global net_apply
    predictions = net_apply(network_params, X)
    return predictions[0]


# Only works with Jax
@maybe_jit
def func_dnn_intrusion_detection(X):
    global network_params
    global net_apply
    predictions = net_apply(network_params, X, rng=key)
    return predictions[0]


############ KL-Divergence ############
@maybe_jit
def func_kld(p_vec):
    epsilon = 1 / (200 * 12)  # sliding_window_size is 200 and num_nodes is 12. Consider using global variables instead of these hard-coded values.

    # Compute KLD (Kullback–Leibler divergence) function.
    # p[:p.shape[0]//2] is the P probability array of the number of observations for each value from 1 to k, and p[p.shape[0]//2:] is the Q probability array.
    P = p_vec[:p_vec.shape[0] // 2]
    Q = p_vec[p_vec.shape[0] // 2:]

    p = P + epsilon
    q = Q + epsilon
    log_p_divide_q = np.log(p / q)
    f = (p * log_p_divide_q).sum()
    return f


############ Inner Product ############
#@maybe_jit
def func_inner_product(X):
    if len(X.shape) >= 2:  # For figures only, not used for grad or Hessian
        x = X[:, :X.shape[1] // 2]
        y = X[:, X.shape[1] // 2:]
        res = np.sum((x * y), axis=1)
        return res

    x = X[:X.shape[0] // 2]
    y = X[X.shape[0] // 2:]

    res = x @ y
    return res


############ Variance ############
#@maybe_jit
def func_variance(X):
    if len(X.shape) >= 2:  # For figures only, not used for grad or Hessian
        return X[:, 1] - X[:, 0] ** 2

    return X[1] - X[0] * X[0]


############ Cosine Similarity ############
@maybe_jit
def func_cosine_similarity(X):

    if len(X.shape) >= 2:  # For figures only, not used for grad or Hessian
        x = X[:, :X.shape[1] // 2]
        y = X[:, X.shape[1] // 2:]
        return np.sum((x * y), axis=1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-10)

    # Use is AutoGrad or Jax as auto gradient tool
    x = X[:X.shape[0] // 2]
    y = X[X.shape[0] // 2:]

    real_denominator = np.linalg.norm(x) * np.linalg.norm(y)
    numerator = np.where(real_denominator != 0, x @ y, 0)
    denominator = np.where(real_denominator != 0, real_denominator, 1)
    res = numerator / denominator
    return res


############ Quadratic Form ############

H = None


def set_H(X_len, H_=None):
    global H
    if H_ is None:
        H = numpy.random.randn(X_len, X_len).astype(np.float32)
    else:
        H = H_.copy()


def get_H():
    global H
    return H


#@maybe_jit
def func_quadratic(X):
    global H

    # Autograd and Jax version of the function
    res = X @ H @ X.T
    return res


############ Rozenbrock ############
#@maybe_jit
def func_rozenbrock(X):
    a = 1.0
    b = 100.0

    if len(X.shape) >= 2:  # For figures only, not used for grad or Hessian
        x = X[:, :X.shape[1] // 2]
        y = X[:, X.shape[1] // 2:]
        res = np.sum((a - x), axis=1) * np.sum((a - x), axis=1) + b * np.sum((y - x * x), axis=1) * np.sum((y - x * x), axis=1)
        return res

    x = X[:X.shape[0] // 2]
    y = X[X.shape[0] // 2:]
    res = (a - x) @ (a - x) + b * (y - x @ x) @ (y - x @ x)
    return res


############ Sine ############
#@maybe_jit
def func_sine(X):
    return np.sin(X)


############ Sine ############
def func_quadratic_inverse(X):
    if len(X.shape) >= 2:  # For figures only, not used for grad or Hessian
        return -X[:, 0]**2 + X[:, 1]**2

    return -X[0]**2 + X[1]**2
