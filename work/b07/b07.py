import torch
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
    b = torch.normal(0.0, 0.01, size=(num_outputs,), generator=g, requries_grad=True)
