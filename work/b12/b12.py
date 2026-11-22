from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path.cwd()


def load_houses(csv_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(f"{csv_dir}/kaggle_house_pred_train.csv")
    test_data = pd.read_csv(f"{csv_dir}/kaggle_house_pred_test.csv")
    return train_data, test_data


def preprocess(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_train = train_data.shape[0]
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    # 去掉Id和SalePrice列，方便模型学习，合并特征，因为后面标准化与one-hot都要用全部数据的统计量
    numeric_features = all_features.select_dtypes(include="number").columns
    # 挑出数值列
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    # 标准化，对每个数值特征减均值，除标准差，apply作用到每一列
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # 用fillna填缺失值，由于标准化后均值为0，所以全填0
    all_features = pd.get_dummies(all_features, dummy_na=True)
    # 独热编码，让缺失值单独成为一个类别
    train_features = torch.tensor(all_features[:n_train].to_numpy(dtype="float32"))
    test_features = torch.tensor(all_features[n_train:].to_numpy(dtype="float32"))
    train_labels = torch.tensor(
        train_data["SalePrice"].values, dtype=torch.float32
    ).reshape(-1, 1)
    # 将读入的数据全部转为张量
    return train_features, test_features, train_labels


def log_rmse(preds: torch.Tensor, labels: torch.Tensor) -> float:
    clipped = torch.clamp(preds, min=1.0)
    return torch.sqrt(((clipped.log() - labels.log()) ** 2).mean()).item()


def get_net(num_features: int = 331):
    return nn.Sequential(nn.Linear(num_features, 1))


def train(
    net: torch.nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor | None,
    test_labels: torch.Tensor | None,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    g = torch.Generator().manual_seed(0)
    train_iter = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss, test_loss = [], []
    for _ in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            loss_fn(net(X), y).backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            train_loss.append(log_rmse(net(train_features), train_labels))
            if test_labels is not None:
                test_loss.append(log_rmse(net(test_features), test_labels))

    return train_loss, test_loss


def get_k_fold_data(
    k: int, i: int, X: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert k > 1, "k需>=1"
    n = X.shape[0]
    base, extra = divmod(n, k)

    sizes = [base + (1 if j < extra else 0) for j in range(k)]
    starts = [sum(sizes[:j]) for j in range(k)]

    parts_x, parts_y = [], []
    for j in range(k):
        idx = slice(starts[j], starts[j] + sizes[j])
        parts_x.append(X[idx])
        parts_y.append(y[idx])

    valid_x, valid_y = parts_x[i], parts_y[i]
    train_x = torch.cat([parts_x[j] for j in range(k) if j != i], dim=0)
    train_y = torch.cat([parts_y[j] for j in range(k) if j != i], dim=0)
    return train_x, train_y, valid_x, valid_y


def k_fold(
    k: int,
    X: torch.Tensor,
    y: torch.Tensor,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> tuple[float, float]:
    train_sum, valid_sum = 0.0, 0.0
    for i in range(k):
        tx, ty, vx, vy = get_k_fold_data(k, i, X, y)
        net = get_net(X.shape[1])
        train_ls, valid_ls = train(
            net, tx, ty, vx, vy, num_epochs, lr, weight_decay, batch_size
        )
        train_sum += train_ls[-1]
        valid_sum += valid_ls[-1]
    return train_sum / k, valid_sum / k


def train_and_pred(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: pd.DataFrame,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> pd.DataFrame:
    net = get_net(train_features.shape[1])
    train_ls, _ = train(
        net,
        train_features,
        train_labels,
        None,
        None,
        num_epochs,
        lr,
        weight_decay,
        batch_size,
    )
    net.eval()
    with torch.no_grad():
        preds = net(test_features).squeeze(1).numpy()

    submission = pd.DataFrame(
        {
            "Id": test_data["Id"].values,
            "SalePrice": preds,
        }
    )
    out_dir = REPO / "work" / "b12"
    out_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_dir / "submission.csv", index=False)
    return submission
