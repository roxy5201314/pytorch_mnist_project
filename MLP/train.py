import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from MLP.model import MLP
from MLP.dataset import get_dataloader
from MLP.utils import evaluate


# 训练主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    epochs = 20

    # ===== 新增：记录 loss 和 accuracy =====
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader, device)

        loss_history.append(avg_loss)
        acc_history.append(acc)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Test Acc: {acc:.4f}"
        )

    # ===== 新增：画图 =====
    epochs_range = range(1, epochs + 1)

    plt.figure()
    plt.plot(epochs_range, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss")
    plt.show()

    plt.figure()
    plt.plot(epochs_range, acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MLP Test Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
