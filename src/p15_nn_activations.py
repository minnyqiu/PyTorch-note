from csv import writer
import os
from pyexpat.errors import XML_ERROR_TAG_MISMATCH
import torch
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter


input = torch.tensor([[1, -0.5],
                      [-1,3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10('PyTorch-note/data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU(inplace=False) # 默认inplace=False
        # input = 1, RuLU(input, inplace=True) --> input = 0
        # input = 1, output = RuLU(input, inplace=False) --> input = 1, output = 0
        self.sigmod1 = Sigmoid()

    def forward(self, input):
        # output = self.relu1(input) 
        output = self.sigmod1(input)
        return output
    
tudui = Tudui()

# output = tudui(input)

# print(output)

log_dir = 'PyTorch-note/logs/p15'

writer = SummaryWriter(log_dir)

# 检查路径是否存在，不存在则创建
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # 创建多级目录
    print(f"目录已创建：{log_dir}")
else:
    print(f"目录已存在：{log_dir}")

step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, global_step=step)
    output = tudui(imgs)
    writer.add_images('output', output, step)
    step = step + 1

writer.close()

# 引入非线性特征
