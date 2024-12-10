import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

dataset = torchvision.datasets.CIFAR10('PyTorch-note/data', train = False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x
    
loss = nn.CrossEntropyLoss()
    
tudui = Tudui()

# 定义优化器
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        # 梯度清零
        optim.zero_grad()
        result_loss.backward() # loss提供梯度
        optim.step() # 调优
        # print(result_loss)
        running_loss = running_loss + result_loss # 每次学习的loss加上
    print(running_loss)
