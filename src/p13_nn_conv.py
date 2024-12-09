import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]]) # torch.Size([5, 5])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]]) # torch.Size([3, 3]) 

# 用reshape使其符合conv2d输入
input = torch.reshape(input, (1,1,5,5)) # batch size, channel, h, w

kernel = torch.reshape(kernel, (1,1,3,3))

print(input.shape)       
print(kernel.shape)   

# stride
output = F.conv2d(input, kernel, stride=1)
print(output) # 3*3

output2 = F.conv2d(input, kernel, stride=2)
print(output2) # 2*2

# padding
output3 = F.conv2d(input, kernel, stride=1, padding=1) # 上下左右填充一行/一列的0
print(output3)

# conv2d, stride = 1 滑动1步 或sH, sW单独设置元组

# 卷积核在输入图像上卷积
# 5*5输入图像 3*3卷积核 得到3*3卷积后的输出 (对应位相乘求和即9个乘积的和)
