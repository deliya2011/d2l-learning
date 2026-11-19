"""
import torch
import torch.nn as nn

def test_acc(net, loader):
    #此处未切到推理模式，精度偏低
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            correct += (net(X).argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.Dropout(0.2),
    nn.ReLU(),#此行位置错误，应与上一行交换，丢弃写在前面会先把一部分线性输出清零
    nn.Linear(256, 10),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5, weight_decay=0.01)#在优化器里添加了weight_decay=0.01,这个实验要比较"有丢弃"与"无丢弃"，权重衰减是另一种正则化手段，
    两者混在一起就分不清精度变化是谁带来的，应去掉。

for epoch in range(5):
    total, n = 0, 0
    for X, y in train_iter:
        optimizer.zero_grad()
        loss = loss_fn(net(X), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * y.numel()
        n += y.numel()
    print(f"epoch {epoch}: 训练损失 {total / n:.4f}  测试精度 {test_acc(net, test_iter):.4f}")
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_high_dim(
    n_train: int = 20, n_test: int = 100, num_inputs: int = 200, seed: int = 0
) -> tuple:
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_test
    x = torch.randn(n, num_inputs, generator=g)
    noise = torch.randn(n, 1, generator=g) * 0.01
    w_true = torch.randn(num_inputs, 1, generator=g) / num_inputs**0.5
    y = 0.05 + x @ w_true + noise
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def l2_penalty(w: torch.Tensor, lam: float) -> torch.Tensor:
    return (lam / 2) * (w**2).sum()


def mean_loss(loader, w, b) -> float:
    total, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            total += ((X @ w + b - y) ** 2).sum().item()
            n += y.numel()
    return total / n


def train_high_dim(
    train_iter,
    test_iter,
    lam: float = 0.0,
    num_epochs: int = 3,
    lr: float = 0.003,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    num_inputs = train_iter.dataset.tensors[0].shape[1]
    w = torch.normal(0, 0.01, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            loss = ((X @ w + b - y) ** 2).sum() + l2_penalty(w, lam) / X.shape[0]
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad = None
                b.grad = None
        train_loss.append(mean_loss(train_iter, w, b))
        test_loss.append(mean_loss(test_iter, w, b))
    return {"train_loss": train_loss, "test_loss": test_loss}


def dropout_layer(X: torch.Tensor, dropout: float) -> torch.Tensor:
    if dropout == 0:
        return X
    elif dropout == 1:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).to(X.dtype)
    return mask * X / (1.0 - dropout)


def make_dropout_net(
    num_inputs: int = 784,
    num_hiddens1: int = 256,
    num_hiddens2: int = 256,
    num_outputs: int = 10,
    dropout1: float = 0.2,
    dropout2: float = 0.5,
    seed: int = 0,
):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(num_hiddens2, num_outputs),
    )


def evaluate(net, loader) -> float:
    was_training = net.training
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            correct += (net(X).argmax(dim=1) == y).sum().item()
            total += y.numel()
    if was_training:
        net.train()
    return correct / total


def train_dropout(
    train_iter,
    test_iter,
    use_dropout: bool = True,
    num_epochs: int = 5,
    lr: float = 0.5,
    batch_size: int = 256,
    seed: int = 0,
) -> dict:
    net = make_dropout_net(
        dropout1=0.2 if use_dropout else 0.0,
        dropout2=0.5 if use_dropout else 0.0,
        seed=seed,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train_loss, train_acc, test_acc = [], [], []
    for epoch in range(num_epochs):
        total, n = 0.0, 0
        for X, y in train_iter:
            optimizer.zero_grad()
            loss = loss_fn(net(X), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * y.numel()
            n += y.numel()
        train_loss.append(total / n)
        train_acc.append(evaluate(net, train_iter))
        test_acc.append(evaluate(net, test_iter))
    return {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}
