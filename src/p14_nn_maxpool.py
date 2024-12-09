import numpy as np
import torch
from torch import nn
from torch.nn import MaxPool2d
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('PyTorch-note/data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]]) # torch.Size([5, 5])
# # print(input.shape)

# input = torch.reshape(input, (-1,1,5,5)) # 用-1自动推断该维度的大小batch size, channel, h, w
# print(input.shape)
# # output = F.max_pool2d(input,kernel_size=3)
# # print(output)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output
    
tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter('PyTorch-note/logs/p14')

step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    output = tudui(imgs)
    writer.add_images('output', output, step)
    step = step + 1

writer.close()

# MaxPool下采样
# 保留输入特征 减少数据量

# kernel_size = 3 即 3*3的池化核(空白的)
# stride 默认值kernel_size
# dilation 每个核元素之间插一个 空洞卷积 (一般设置)
# ceil_mode默认False (floor向下取整 ceil向上取整)

# 取池化核覆盖的最大值输出
# 根据stride=kernel_size移动
# 当池化核移动到没有数值的位置ceil_mode=True保留该位置的output
