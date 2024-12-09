import torch
from torch import nn

# containers

# input output

# 神经网络模板
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    # 输出只给输入加一
    def forward(self, input):
        output = input + 1
        return  output

# 用模板创建的神经网络
tudui = Tudui()

x = torch.tensor(1.0)
output = tudui(x) # x输入放到神经网络
print(output)

