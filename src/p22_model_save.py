import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights=None) # 仅加载模型参数 不训练
# print(vgg16)

# # 保存方式1 - 模型结构 + 模型参数
torch.save(vgg16, 'PyTorch-note/output/vgg16_method1.pth') # 结构和参数都保存

# 保存方式2 - 模型参数 (状态保存成字典) [官方推荐]
torch.save(vgg16.state_dict(), 'PyTorch-note/output/vgg16_method2.pth')

# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x
    
tudui = Tudui()

torch.save(tudui, 'PyTorch-note/output/tudui.pth')
