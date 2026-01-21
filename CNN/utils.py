import torch

# 评估函数
def evaluate(model, dataloader, device):
    model.eval() # 切换到评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item() # 统计正确预测的数量
            total += y.size(0)

    acc = correct / total
    model.train()
    return acc
