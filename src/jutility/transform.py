import numpy as np

class Linear:
    def __init__(self, w):
        self.w = w

    def __call__(self, x):
        return self.w @ x

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, x):
        return self.w @ x + self.b

def outer_product_batched(x, y):
    nx_0, nx_1 = x.shape
    ny_0, ny_1 = y.shape
    p = x.reshape(nx_0, 1, nx_1) * y.reshape(1, ny_0, ny_1)
    return p

def least_squares_linear(x, y, reg=1e-5, batch_first=False):
    if batch_first:
        x = x.T
        y = y.T

    nx, nd = x.shape
    cov_xx = outer_product_batched(x, x).mean(axis=2)
    cov_yx = outer_product_batched(y, x).mean(axis=2)
    w = cov_yx @ np.linalg.inv(cov_xx + reg * np.identity(nx))
    f = Linear(w)
    return f

def least_squares_affine(x, y, reg=1e-5, batch_first=False):
    if batch_first:
        x = x.T
        y = y.T

    nx, nd = x.shape
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    x_zero_mean = x - x_mean
    y_zero_mean = y - y_mean
    cov_xx = outer_product_batched(x_zero_mean, x_zero_mean).mean(axis=2)
    cov_yx = outer_product_batched(y_zero_mean, x_zero_mean).mean(axis=2)
    w = cov_yx @ np.linalg.inv(cov_xx + reg * np.identity(nx))
    b = y_mean - w @ x_mean
    f = Affine(w, b)
    return f
