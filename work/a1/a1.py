import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        # 此处缺少super().__init__()，未调用父类构造函数
        self.weight = torch.randn(in_dim, out_dim)
        self.bias = torch.zeros(out_dim)  # 这两行都没有使用nn.Parameter(..)，没有登记

    def forward(self, X):
        return X @ self.weight + self.bias  # 返回值错误，应为self.weight+self.bias


def batches(
    data=[],
):  # 此行错误，因为这样调用函数会使data值堆积，应该将data=[]改为data=None
    for i in range(len(data)):
        return data[i]


net = Net(3, 1)
print(len(list(net.parameters())))
print(net(torch.randn(2, 3)))
