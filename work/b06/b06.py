"""
import torch
import torch.nn as nn

w = torch.randn(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr, batch_size = 0.5, 10
net = lambda X: X @ w + b
loss_fn = lambda y_hat, y: (y_hat - y) ** 2

for epoch in range(3):
    for X, y in batches:
        l = loss_fn(net(X), y) #此行错误，这里的l是一个张量，而下一行的l.backward()针对非标量会报错，应改为l=loss_fn(net(X),y).mean()
        l.backward() #此行错误，梯度会错误累加，不会清零，导致每个batch都在用所有历史batch的梯度之和更新，应该添加w.grad.zero_()和b.grad.zero_()
        w = w - lr * w.grad / batch_size #此行及下一行错误，因为这样写会创造一个新的张量赋给w和b，且新w的.grad是None，下一次循环会报错，应在原地更新，使用with torch.no_grad():
        b = b - lr * b.grad / batch_size
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset


def make_data(
    n: int, w: torch.Tensor, b: float, noise: float = 0.01, seed: int = 0
) -> tuple:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, len(w), generator=g, dtype=torch.float32)
    eps = torch.normal(0.0, 0.01, size=(n, 1), generator=g)
    w = w.to(torch.float32).reshape(-1, 1)
    y = x @ w + b
    eps = torch.normal(0.0, noise, size=(n, 1), generator=g, dtype=torch.float32)
    y = y + eps
    return x, y


def data_iter(
    batch_size: int, features: torch.Tensor, labels: torch.Tensor, seed: int = 0
):
    n = features.shape[0]
    g = torch.Generator().manual_seed(seed)
    divide = torch.randperm(n, generator=g)
    for start in range(0, n, batch_size):
        batch_idx = divide[start : start + batch_size]
        yield features[batch_idx], labels[batch_idx]


def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return X @ w + b


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.reshape(y_hat.shape)
    return (y_hat - y) ** 2 / 2


def sgd(params: list, lr: float, batch_size: int) -> None:
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()


def train_scratch(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.03,
    num_epochs: int = 3,
    batch_size: int = 10,
    seed: int = 0,
) -> tuple:
    d = X.shape[1]
    g = torch.Generator().manual_seed(seed)
    w = torch.normal(0.0, 0.01, size=(d, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    losses = []
    for epoch in range(num_epochs):
        for x1, y1 in data_iter(batch_size, X, y, seed + epoch):
            l = squared_loss(linreg(x1, w, b), y1)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = squared_loss(linreg(X, w, b), y).mean()
        losses.append(train_l.item())

    return w, b, losses


def train_concise(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.03,
    num_epochs: int = 3,
    batch_size: int = 10,
    seed: int = 0,
) -> tuple:
    torch.manual_seed(seed)

    net = nn.Sequential(nn.Linear(X.shape[1], 1))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        for Xb, yb in loader:
            l = loss_fn(net(Xb), yb)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        with torch.no_grad():
            train_l = loss_fn(net(X), y)
        losses.append(train_l.item())

    return net, losses
