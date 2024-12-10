import torch
from torch import nn
import torchvision
from torchvision.models import VGG16_Weights

# train_data = torchvision.datasets.ImageNet('PyTorch-note/data',split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

# vgg16_false = torchvision.models.vgg16(pretrained=False)
# vgg16_true = torchvision.models.vgg16(pretrained=True)

vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
vgg16_false = torchvision.models.vgg16(weights=None) # 仅加载模型

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('PyTorch-note/data', transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 用现有的网络改动结构, VGG16最后一层(6): Linear(in_features=4096, out_features=1000, bias=True)
# CIFAR10 只有10类数据

# 1.添加一层线性层
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10)) # 加classifier 把添加的层加到最后的classifier里
print(vgg16_true) #   (add_linear): Linear(in_features=1000, out_features=10, bias=True)

print(vgg16_false)

# 或

# 2.直接修改最后一层
vgg16_false.classifier[6] = nn.Linear(4096, 10) 
print(vgg16_false) # (6): Linear(in_features=4096, out_features=10, bias=True)
