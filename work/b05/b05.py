"""
import torch

# 假设事件 A 与事件 B 相互独立
p_a = 0.3
p_b = 0.4

# 下面这行想算出 P(A 且 B)，也就是联合概率
p_a_and_b = p_a + p_b #此行错误，这是P(a|b)的公式，应改为p_a_and_b = p_a * p_b

counts = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
probs = counts #此行错误，counts的这几个数的总和不为1，不是概率，应该改为：probs = counts / counts.sum()

values = torch.tensor([1.0, 2.0, 3.0])
weights = torch.tensor([0.2, 0.3, 0.5])
E = values.mean() #期望值 E 算的是"无权平均"，忽略了 weights ,应该改为：E = (values * weights).sum()
"""

import torch
from torch.distributions import multinomial


def roll_and_estimate(n_rolls: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    probs = torch.full((6,), 1 / 6)
    rolls = torch.multinomial(probs, n_rolls, replacement=True, generator=g)
    counts = torch.bincount(rolls, minlength=6).float()
    return counts / n_rolls


def mean_var(values: torch.Tensor, probs: torch.Tensor) -> tuple:
    expected_value = (values * probs).sum()
    var = (values**2 * probs).sum() - expected_value**2
    return expected_value, var


def posterior(p_d_given_h1: float, p_d_given_h0: float, p_h1: float) -> float:
    p_d1 = p_d_given_h0 * (1 - p_h1) + p_d_given_h1 * p_h1
    p_d1_given_h1 = (p_d_given_h1 * p_h1) / p_d1
    return p_d1_given_h1


def cumulative_means(n_max: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    rolls = torch.randint(1, 7, (n_max,), generator=g, dtype=torch.float32)
    return torch.cumsum(rolls, dim=0) / torch.arange(1, n_max + 1, dtype=torch.float32)
