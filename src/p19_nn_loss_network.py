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

for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    # 1.计算实际输出和目标之间的差距
    result_loss = loss(outputs, targets)
    # print(outputs)
    # print(targets)
    # print(result_loss)

    # 2.为更新输出提供一定的依据（反向传播）grad
    result_loss.backward() # debug 看grad - tudui - model1 - _modules - 0 - Continue
    print('ok')
