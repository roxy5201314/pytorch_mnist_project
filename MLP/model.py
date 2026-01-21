import torch.nn as nn

# 模型定义
# 基础多层感知机(MLP) multi-layer perceptron
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 神经网络结构
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(), #激活函数 线性->非线性
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
