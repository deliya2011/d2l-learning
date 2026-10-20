def grad_of_quadratic(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (A + A.T) @ x
