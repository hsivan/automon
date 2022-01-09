import jax.numpy as np  # the user could use autograd.numpy instead of JAX (for Windows)


def func_inner_product(x):
    return np.matmul(x[:x.shape[0] // 2], x[x.shape[0] // 2:])
