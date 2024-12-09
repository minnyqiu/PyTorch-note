import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10('PyTorch-note/data', train = False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

# 线性层网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10) # in_features, out_features

    def forward(self, input):
        output = self.linear1(input)
        return output
    
tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape) # torch.Size([64, 3, 32, 32]) <--(batch_size, channels, height, width)
    # output = torch.reshape(imgs,(1,1,1,-1))  
    # print(output.shape) # reshape: torch.Size([1, 1, 1, 196608]) <-- total_elements=64×3×28×28=150528 仍然是一个 4 维张量，形状是 (1, 1, 1, total_elements)
    output = torch.flatten(imgs) # 不用reshape 换成flatten
    print(output.shape) #  flatten: torch.Size([196608])
    output = tudui(output)
    print(output.shape)  # reshape: torch.Size([1, 1, 1, 10]) 四维 / flatten: torch.Size([10]) 一维

# Liner 线性层

# 5*5的图片 转成一行25格
