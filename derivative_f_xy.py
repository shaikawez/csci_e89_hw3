"""Partial derivatives of f(x, y) = x^2 + y^2 + 2xy at the point (2, 3).

Computed three ways: analytically, with PyTorch autograd, and with
central finite differences as a numerical check.
"""
import torch


def f(x, y):
    return x**2 + y**2 + 2 * x * y


x0, y0 = 2.0, 3.0

# 1. Analytic: df/dx = 2x + 2y, df/dy = 2y + 2x
dfdx_exact = 2 * x0 + 2 * y0
dfdy_exact = 2 * y0 + 2 * x0

# 2. PyTorch autograd
x = torch.tensor(x0, requires_grad=True)
y = torch.tensor(y0, requires_grad=True)
z = f(x, y)
z.backward()

# 3. Central finite differences
h = 1e-5
dfdx_num = (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h)
dfdy_num = (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)

print(f"f(2, 3)          = {z.item()}")
print(f"Analytic   grad  = ({dfdx_exact}, {dfdy_exact})")
print(f"Autograd   grad  = ({x.grad.item()}, {y.grad.item()})")
print(f"Numerical  grad  = ({dfdx_num:.6f}, {dfdy_num:.6f})")
