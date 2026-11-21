"""
import torch

def relu(t):
    return torch.maximum(t, torch.zeros_like(t))

def backward(x, W1, b1, W2, b2, y, lam=0.0):
    z = W1 @ x + b1
    h = relu(z)
    o = W2 @ h + b2
    L = 0.5 * ((o - y) ** 2).sum()
    s = lam / 2 * ((W1 ** 2).sum() + (W2 ** 2).sum())
    J = L + s

    d_o = o - y                          # 对 o 的梯度
    d_W2 = d_o @ h.t() + lam * W2        # 对 W2 的梯度
    d_b2 = d_o.sum()                     # 对 b2 的梯度 #此处错误，偏置b2的形状是(q,1)，而d_o.sum()是标量，应添加dim=1,keepdim=True
    d_h = W2 @ d_o                       # 对 h 的梯度 #此处错误，W2未转置，应改为W2.T@d_o
    d_z = d_h * (z >= 0).float()         # 对 z 的梯度 #此处错误，relu等于0时不可导,应改为d_h * (z > 0).float()
    d_W1 = d_z @ x.t() + lam * W1        # 对 W1 的梯度
    d_b1 = d_z.sum()                     # 对 b1 的梯度
    return J, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
"""

import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.zeros_like(x))


def relu_grad(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).float()


def forward_backward(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    y: torch.Tensor,
    lam: float = 0.0,
) -> tuple:
    z = W1 @ x + b1
    h = relu(z)
    o = W2 @ h + b2
    L = 0.5 * ((o - y) ** 2).sum()
    s = lam / 2 * ((W1**2).sum() + (W2**2).sum())
    J = L + s
    d_o = o - y
    d_W2 = d_o @ h.T + lam * W2
    d_b2 = d_o
    d_h = W2.T @ d_o
    d_z = d_h * relu_grad(z)
    d_W1 = d_z @ x.T + lam * W1
    d_b1 = d_z
    return J, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}


def numeric_grad(f, x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    grad = torch.zeros_like(x)
    x_flat = x.view(-1)
    g_flat = grad.view(-1)

    for i in range(x_flat.numel()):
        orig = x_flat[i].item()

        x_flat[i] = orig + eps
        f_plus = f(x)

        x_flat[i] = orig - eps
        f_minus = f(x)

        x_flat[i] = orig
        g_flat[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def grad_check(
    analytic: torch.Tensor, numeric: torch.Tensor, tol: float = 1e-4
) -> bool:
    diff = (analytic - numeric).norm()
    denom = (analytic.norm() + numeric.norm()).clamp_min(1e-8)
    rel = diff / denom
    return bool(rel < tol)


def xavier_uniform(a: int, b: int, seed: int = 0) -> torch.Tensor:
    r = (6 / (a + b)) ** 0.5
    g = torch.Generator().manual_seed(seed)
    return torch.rand(a, b, generator=g) * 2 * r - r


def he_uniform(a: int, b: int, seed: int = 0) -> torch.Tensor:
    r = (6 / a) ** 0.5
    g = torch.Generator().manual_seed(seed)
    return torch.rand(a, b, generator=g) * 2 * r - r
