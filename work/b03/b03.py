"""
import torch

A = torch.arange(6.0).reshape(2, 3)
x = torch.ones(3)

# 想算 A 乘以向量 x
y = A * x #此行错误，A*x是点积的写法，即相同位置元素乘积之和，矩阵乘向量应该用torch.mv(A,x)

# 想每一行除以该行元素之和
s = A.sum(dim=1) #此行错误，sum会导致被指定的轴消失，应改为s=A.sum(dim=1,keepdim=True)
z = A / s

# 想交换 B 的最后两维
B = torch.arange(24.0).reshape(2, 3, 4)
C = B.T #B是一个三维张量，应用B.transpose(1,2)
"""

import torch


def reduce_shape(shape: tuple, dim: int, keepdim: bool) -> tuple:
    dim = dim % len(shape)
    out = list(shape)
    if keepdim:
        out[dim] = 1
    else:
        del out[dim]
    return tuple(out)


def matmul_kind(a: torch.Tensor, b: torch.Tensor) -> str:
    if len(a.shape) == len(b.shape) and len(a.shape) == 1:
        return "dot"
    elif (len(a.shape) == 1 and len(b.shape) == 2) or (
        len(a.shape) == 2 and len(b.shape) == 1
    ):
        return "mv"
    elif len(a.shape) == 2 and len(b.shape) == 2:
        return "mm"
    else:
        return "matmul"


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1)
    x = x**2
    x = torch.sqrt(x.sum(dim=0))
    return x


def normalize_rows(X: torch.Tensor) -> torch.Tensor:
    norms = X.norm(p=2, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return X / norms
