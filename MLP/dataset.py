from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 加载数据
def get_dataloader(batch_size=64, train=True):
    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return dataloader
