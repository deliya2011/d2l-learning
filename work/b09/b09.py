import torch


def make_data(n_train: int = 100, n_test: int = 100, seed: int = 0) -> tuple:
    g = torch.Generator().manual_seed(seed)
    x_train = torch.randn(n_train, 1, generator=g, dtype=torch.float32)
    noise = torch.normal(0, 0.1, size=(1,), generator=g, dtype=torch.float32)
    y_train = 1.2 @ x_train - 3.4 @ (x_train**2) + 5.6 @ (x_train**3) + 5 + noise
    x_test = torch.randn(n_test, 1, generator=g, dtype=torch.float32)
    y_test = 1.2 @ x_test - 3.4 @ (x_test**2) + 5.6 @ (x_test**3) + 5 + noise
    return x_train, y_train, x_test, y_test
