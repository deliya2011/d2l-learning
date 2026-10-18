"""
import torch

# 造一批数据，准备拿去做浮点运算
x = torch.arange(12) #此行错误，因为张量的默认类型为int64，应该写为x=torch.arange(12,dtype=torch.float32)

# 变形成 3 行 4 列
X = x.reshape(3, 4)

# 打印它的形状
print(X.size) #此行错误，因为size是方法，需要()，应该写为X.size()

# 取第一行改一改，认为 X 不会受影响
# 上述作者的预期是X不会受影响，但是实际上X已经受到影响，第一行变为了100.0
row = X[0]
row[0] = 100.0
print(X[0, 0])
"""

import torch
import pandas as pd


def tensor_info(x: torch.Tensor) -> dict:
    return {
        "shape": tuple(x.shape),
        "numel": x.numel(),
        "dtype": str(x.dtype).removeprefix("torch."),
        "device": str(x.device),
        "is_contiguous": x.is_contiguous(),
    }


def safe_broadcast_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sa = tuple(a.shape)
    sb = tuple(b.shape)
    max_len = max(len(sa), len(sb))
    a1 = (1,) * (max_len - len(sa)) + sa
    b1 = (1,) * (max_len - len(sb)) + sb
    for i, (a2, b2) in enumerate(zip(a1, b1)):
        if a2 != b2 and a2 != 1 and b2 != 1:
            raise ValueError(f"第 {i} 维：{a2} 与 {b2}")
    return a + b


def slice_is_view() -> bool:
    x = torch.arange(12)
    y = x[1:3]
    y[0] = 114514
    return bool(x[1].item() == 114514)


def clone_is_copy() -> bool:
    x = torch.arange(12)
    y = x[1:3].clone()
    y[0] = 114514
    return bool(x[1].item() == 114514)


# def load_csv(path: str) -> tuple[torch.Tensor, list[str], dict]:
