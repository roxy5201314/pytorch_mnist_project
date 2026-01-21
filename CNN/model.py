import torch.nn as nn

# 卷积神经网络(CNN) Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积特征提取
        self.features = nn.Sequential(
            # 卷积层
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 32, 28, 28]
            nn.ReLU(),
            # 池化层
            nn.MaxPool2d(2),                             # [B, 32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2)                              # [B, 64, 7, 7]
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x
