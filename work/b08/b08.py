"""
import torch
import torch.nn as nn

torch.manual_seed(0)
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.Linear(256, 10),#两个nn.Linear间没有激活函数，与下一个错误改法相同
    nn.ReLU(),#此处错误。ReLU位置应该和上一行交换。
)
for name, p in net.named_parameters():
    with torch.no_grad():
        p.zero_()#此处错误，p不能全0初始化，会让所有隐藏单元永远保持相同，应在权重上使用均值为0，标准差为0.01的正态分布随机数，偏置可全0

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(10):
    for X, y in train_iter:
        optimizer.zero_grad()
        l = loss_fn(net(X), y)
        l.backward()
        optimizer.step()
"""

import torch
import torch.nn as nn


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.zeros_like(x))


def relu_grad(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).float()


def init_params(
    num_inputs: int = 784, num_hiddens: int = 256, num_outputs: int = 10, seed: int = 0
) -> tuple:
    g = torch.Generator().manual_seed(seed)
    W1 = torch.normal(
        0.0, 0.01, size=(num_inputs, num_hiddens), generator=g, requires_grad=True
    )
    b1 = torch.zeros(num_hiddens, requires_grad=True)
    W2 = torch.normal(
        0.0, 0.01, size=(num_hiddens, num_outputs), generator=g, requires_grad=True
    )
    b2 = torch.zeros(num_outputs, requires_grad=True)
    return W1, b1, W2, b2


def mlp_forward(
    X: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    X = X.reshape(X.shape[0], -1)
    H = relu(X @ W1 + b1)
    O = H @ W2 + b2
    return O


def count_params(net) -> int:
    return sum(p.numel() for p in net.parameters())


def sgd(params: list, lr: float, batch_size: int) -> None:
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()


def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = logits.max(dim=1).values
    logsumexp = m + torch.log(torch.exp(logits - m.unsqueeze(1)).sum(dim=1))
    correct_y = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    loss = (logsumexp - correct_y).sum()
    return loss


def evaluate_accuracy(net, loader) -> float:
    acc = 0
    tot = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.reshape(X.shape[0], -1)
            pred = net(X).argmax(dim=1)
            acc += (pred == y).sum().item()
            tot += y.numel()
    return acc / tot


def train_from_scratch(
    train_iter,
    test_iter,
    num_epochs: int = 10,
    lr: float = 0.1,
    batch_size: int = 256,
    seed: int = 0,
) -> dict:
    W1, b1, W2, b2 = init_params(seed=seed)
    net = lambda x: mlp_forward(x, W1, b1, W2, b2)
    train_loss, train_acc, test_acc = [], [], []
    for epoch in range(num_epochs):
        tot = 0.0
        n_samples = 0
        for X, y in train_iter:
            X = X.reshape(X.shape[0], -1)
            l = cross_entropy(net(X), y)
            l.backward()
            sgd([W1, b1, W2, b2], lr, batch_size)
            tot += l.item()
            n_samples += y.numel()
        train_loss.append(tot / n_samples)
        train_acc.append(evaluate_accuracy(net, train_iter))
        test_acc.append(evaluate_accuracy(net, test_iter))
    return {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}


def train_concise(
    train_iter, test_iter, num_epochs: int = 10, lr: float = 0.1, seed: int = 0
) -> dict:
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train_loss, train_acc, test_acc = [], [], []
    for epoch in range(num_epochs):
        tot = 0
        n_samples = 0
        for X, y in train_iter:
            X = X.reshape(X.shape[0], -1)
            l = loss_fn(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            tot += l.item()
            n_samples += y.numel()
        train_loss.append(tot / n_samples)
        train_acc.append(evaluate_accuracy(net, train_iter))
        test_acc.append(evaluate_accuracy(net, test_iter))
    return {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}
