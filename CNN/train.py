import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import CNN
from dataset import get_dataloader
from utils import evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
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

    # ===== 自动画图 =====
    epochs_range = range(1, epochs + 1)

    plt.figure()
    plt.plot(epochs_range, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (CNN)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(epochs_range, acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy Curve (CNN)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
