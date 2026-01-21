import torch
import torch.nn as nn
import torch.optim as optim

from model import MLP
from dataset import get_dataloader
from utils import evaluate

# 训练主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.1) # 随机梯度下降优化器

    epochs = 20 # 训练轮数

    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad() # 梯度清零
            outputs = model(x) # 前向传播
            loss = criterion(outputs, y) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

        acc = evaluate(model, test_loader, device)
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f} "
            f"Test Acc: {acc:.4f}"
        )


if __name__ == "__main__":
    main()
