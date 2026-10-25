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
