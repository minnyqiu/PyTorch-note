from turtle import forward
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

# 写CIFAR10 模型

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2) # in, out, nernel, padding根据文档公式倒推
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2) # in, out, nernel 
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2) # in, out, nernel 
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)
        # 引入Sequential 代码更简洁
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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # # x = self.linear1(x) # 注释掉直接让网络计算 得到线性层的input
        # # x = self.linear2(x)
        x = self.model1(x) # 通过Sequential依次经过手写的forward
        return x

tudui = Tudui()
print(tudui)

# 检验网络正确性
input = torch.ones((64, 3, 32, 32))

output = tudui(input)

print(output.shape)

# 用tensorboard可视化
log_dir = 'PyTorch-note/logs/p17'

writer = SummaryWriter(log_dir)
writer.add_graph(tudui, input)
writer.close()
