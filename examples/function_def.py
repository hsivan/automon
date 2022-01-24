import jax.numpy as np  # For Windows, import autograd.numpy instead of jax.numpy


def func_inner_product(x):
    return np.matmul(x[:x.shape[0] // 2], x[x.shape[0] // 2:])
