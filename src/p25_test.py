from PIL import Image
from numpy import argmax
import torch
import torchvision
from torch import nn

image_path = 'PyTorch-note/data/imgs/dog.png'

img = Image.open(image_path)

print(img) # Image没有shape AttributeError: 'JpegImageFile' object has no attribute 'shape'

img = img.convert('RGB') # 确保RGB三通道

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), # 注意这里的括号 两层
                                            torchvision.transforms.ToTensor()])

img = transform(img)

print(img.shape) # torch.Size([3, 32, 32]) 变成tensor所以有shape

# 根据p24保存模型的方式 加载模型

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64,10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load('PyTorch-note/output/tudui_19_gpu.pth', weights_only=False)

# print(model)

img = torch.reshape(img, (1, 3, 32, 32))

if torch.cuda.is_available():
    img = img.cuda()  # 将输入数据转移到 GPU

model.eval()

with torch.no_grad():
    output = model(img)

print(output) 
# tensor([[-0.0711, -4.3447,  3.7525,  1.4279,  
# 3.9666,  2.5298,  1.8437,  0.6046, -3.9310, -4.1440]], device='cuda:0')

print(output.argmax(1)) # 行里面最大的
# tensor([4], device='cuda:0')
# 第4类最大 通过debug查看标签对应的类别 

'''
完整的模型验证(测试 demo)套路
利用已经训练好的模型
向其提供输入
'''