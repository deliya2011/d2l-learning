import torch

"""
import torch

x = torch.tensor([1.0, 2.0, 3.0]) #此处错误，非叶子张量x.grad为None，在y.backward()中会报错，应改为x=torch.tensor([1.0, 2.0, 3.0],requires_gard=True) 
y = (x * 2).sum()
y.backward()
print(x.grad)

w = torch.tensor([1.0, 2.0])
z = w * 3
z.backward() #此处错误，非标量输出调用backward必须给gradient，否则会报`RuntimeError: grad can be implicitly created only for scalar outputs`，应改为z.backward(torch.ones_like(y))

p = torch.tensor([1.0], requires_grad=True)
y = (p * 2).sum()
y.backward()
p = p - 0.1 * p.grad #此处错误，应写为p-=0.1*p.grad
"""


def grad_of_quadratic(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (A + A.T) @ x


def grad_report(x: torch.Tensor) -> dict:
    y = (x * 2).sum()
    y.backward()
    return {
        "x_grad": x.grad,
        "y_grad": y.grad,
        "x_is_leaf": x.is_leaf,
        "y_is_leaf": y.is_leaf,
    }


def grad_after_two_backwards(x: torch.Tensor) -> torch.Tensor:
    y = (x * x).sum()
    y.backward(retain_graph=True)
    y.backward()
    return x.grad


def descend_step(w: torch.Tensor, step: float) -> None:
    y = (w * w).sum()
    y.backward()
    with torch.no_grad():
        w -= step * w.grad
    w.grad.zero_()
