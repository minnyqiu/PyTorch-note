import torch
import torchvision
from torch import nn
from p22_model_save import *

# # 加载模型 - 保存方式1
# model = torch.load('PyTorch-note/output/vgg16_method1.pth')
# print(model)

# # 加载模型 - 保存方式2
# # 新建模型结构
# vgg16 = torchvision.models.vgg16(weights=None)
# vgg16.load_state_dict(torch.load('PyTorch-note/output/vgg16_method2.pth', weights_only=True))
# # print(model))
# # model = torch.load('PyTorch-note/output/vgg16_method2.pth')
# print(vgg16)

# 陷阱 使用保存方式1
# Can't get attribute 'Tudui'
# 需要把网络结构复制过来 (gyy-motion中遇到的问题)

# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

#     def forward(self, x):
#         x = self.conv1(x)
#         return x

tudui = torch.load('PyTorch-note/output/tudui.pth', weights_only=False) 
print(tudui)