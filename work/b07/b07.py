"""
import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(784, 10))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(5):
    for X, y in train_iter:
        X = X.reshape(X.shape[0], -1)
        out = torch.softmax(net(X), dim=1) #此处错误，相当于做了两次softmaxnet，因为nn.CrossEntropyLoss里已经做了一次softmax，应删掉此行
        l = loss_fn(out, y)
        optimizer.step()
        l.backward() #此处错误，优化器应该在梯度更新之后再更新，应交换此行与上行位置
        #第三个错误，整体没有清零梯度，应在l.backward()前添加optimizer.zero_grad()
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def softmax(o: torch.Tensor) -> torch.Tensor:
    m = o.max(dim=-1, keepdim=True).values
    e = torch.exp(o - m)
    return e / e.sum(dim=-1, keepdim=True)


def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = logits.max(dim=1).values
    logsumexp = m + torch.log(torch.exp(logits - m.unsqueeze(1)).sum(dim=1))
    correct_y = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    loss = (logsumexp - correct_y).sum()
    return loss


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


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


def sgd(params: list, lr: float, batch_size: int) -> None:
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()


def make_net(num_inputs: int = 784, num_outputs: int = 10, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    W = torch.normal(
        0.0, 0.01, size=(num_inputs, num_outputs), generator=g, requires_grad=True
    )
    b = torch.zeros(num_outputs, requires_grad=True)
    return W, b


def train_from_scratch(
    train_iter,
    test_iter,
    num_epochs: int = 5,
    lr: float = 0.1,
    batch_size: int = 256,
    seed: int = 0,
) -> dict:
    W, b = make_net(seed=seed)
    net = lambda x: x @ W + b
    train_loss, train_acc, test_acc = [], [], []
    for epoch in range(num_epochs):
        tot = 0
        n_samples = 0
        for X, y in train_iter:
            X = X.reshape(X.shape[0], -1)
            l = cross_entropy(net(X), y)
            l.backward()
            sgd([W, b], lr, batch_size)
            tot += l.item() * y.numel()
            n_samples += y.numel()
        train_loss.append(tot / n_samples)
        train_acc.append(evaluate_accuracy(net, train_iter))
        test_acc.append(evaluate_accuracy(net, test_iter))
    return {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}


def confusion_matrix(net, loader, q: int = 10) -> torch.Tensor:
    xter = torch.zeros(q, q, dtype=torch.long)
    with torch.no_grad():
        for X, y in loader:
            X = X.reshape(X.shape[0], -1)
            pred = net(X).argmax(dim=1)
            for t, p in zip(y.tolist(), pred.tolist()):
                xter[t, p] += 1
    return xter


def train_concise(
    train_iter, test_iter, num_epochs: int = 5, lr: float = 0.1, seed: int = 0
) -> dict:
    net = nn.Sequential(nn.Linear(784, 10))
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
            tot += l.item() * y.numel()
            n_samples += y.numel()
        train_loss.append(tot / n_samples)
        train_acc.append(evaluate_accuracy(net, train_iter))
        test_acc.append(evaluate_accuracy(net, test_iter))
    return {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}
