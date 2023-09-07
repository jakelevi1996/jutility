from typing import Any
import numpy as np

class LinearTransform:
    def __init__(self, w, batch_first=False):
        self.w = w

    def __call__(self, x):
        return self.w @ x

class AffineTransform:
    def __init__(self, w, b, batch_first=False):
        self.w = w
        self.b = b

    def __call__(self, x):
        return self.w @ x + self.b

def outer_product_batched(x, y):
    nx, nd = x.shape
    ny, nd = y.shape
    p = x.reshape(nx, 1, nd) * y.reshape(1, ny, nd)
    return p

def least_squares_affine_transform(x, y, reg=1e-5, batch_first=False):
    nx, nd = x.shape
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    x_zero_mean = x - x_mean
    y_zero_mean = y - y_mean
    cov_xx = outer_product_batched(x_zero_mean, x_zero_mean).mean(axis=2)
    cov_yx = outer_product_batched(y_zero_mean, x_zero_mean).mean(axis=2)
    w = cov_yx @ np.linalg.inv(cov_xx + reg * np.identity(nx))
    b = y_mean - w @ x_mean
    return AffineTransform(w, b, batch_first)
