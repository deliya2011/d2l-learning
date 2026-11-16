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


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.reshape(y_hat.shape)
    return (y_hat - y) ** 2 / 2


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
