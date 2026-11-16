"""
import matplotlib.pyplot as plt
import torch

w = torch.zeros(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
train_curve, test_curve = [], []

for epoch in range(100):
    for X, y in train_iter:
        pred = X @ w + b
        loss = ((pred - y) ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            #没有清零，梯度会一直累积，导致训练发散，应添加w.grad=None和b.grad=None
    train_curve.append(loss.item())#此行错误，训练时返回和没有问题，但是此处应该返回平均(loss/X.shape[0]).item()
    test_curve.append(((X_test @ w + b - y_test) ** 2).sum())

plt.plot(train_curve, test_curve) #此行错误，把训练误差当成了横轴，测试误差当成了纵轴，画出来的是两条误差的关系图，以及并没有savefig
plt.show()
"""

import torch
import matplotlib
import matplotlib.pyplot as plt


def make_data(n_train: int = 100, n_test: int = 100, seed: int = 0) -> tuple:
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_test
    x = torch.randn(n, 1, generator=g)
    noise = torch.randn(n, 1, generator=g) * 0.1
    y = 1.2 * x - 3.4 * x**2 + 5.6 * x**3 + 5.0 + noise
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def poly_features(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.cat([x**i for i in range(1, k + 1)], dim=1)


def squared_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred - y) ** 2).sum()


def sgd(params: list, lr: float, batch_size: int) -> None:
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()


def mean_loss(pred: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return ((pred - y) ** 2).mean().item()


def train_poly(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    k: int,
    num_epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 10,
    seed: int = 0,
) -> dict:
    g = torch.Generator().manual_seed(seed)
    X_train = poly_features(x_train, k)
    X_test = poly_features(x_test, k)
    w = torch.zeros(k, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    n = X_train.shape[0]
    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        order = torch.randperm(n, generator=g)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            loss = squared_loss(X_train[idx] @ w + b, y_train[idx])
            loss.backward()
            sgd([w, b], lr, idx.numel())
        train_loss.append(mean_loss(X_train @ w + b, y_train))
        test_loss.append(mean_loss(X_test @ w + b, y_test))
    return {"train_loss": train_loss, "test_loss": test_loss}


def plot_errors(train_loss: list, test_loss: list, path: str) -> None:
    plt.figure(figsize=(7, 4))
    epochs = range(1, len(train_loss) + 1)
    plt.semilogy(epochs, train_loss, label="training error")
    plt.semilogy(epochs, test_loss, label="generalization error")
    plt.xlabel("epoch")
    plt.ylabel("mean squared error(log scale)")
    plt.legend()
    plt.savefig(path, dpi=120)


def k_fold_indices(n: int, k: int, seed: int = 0) -> list:
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()
    base = n // k
    extra = n % k
    parts, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        parts.append(sorted(order[start : start + size]))
        start += size
    return parts


def diagnose(train_loss: list, test_loss: list) -> str:
    if train_loss[-1] > 0.5:
        return "underfitting"
    if test_loss[-1] > train_loss[-1] * 10:
        return "overfitting"
    return "good"
