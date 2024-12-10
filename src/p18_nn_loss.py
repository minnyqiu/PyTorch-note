import torch
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs,(1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = nn.L1Loss(reduction='sum') # 计算方式默认mean
result = loss(inputs, targets)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print(result) # tensor(0.6667) / tensor(2.)
print(result_mse) # tensor(1.3333)

x = torch.tensor([0.1, 0.2, 0.3]) # probabilities of each class
print(x.shape) # torch.Size([3])
y = torch.tensor([1]) # target class
x = torch.reshape(x, (1, 3))
print(x.shape) # torch.Size([1, 3])
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross) # tensor(1.1019)

# 1.计算实际输出和目标之间的差距
# 2.为更新输出提供一定的依据（反向传播）grad

# 分类 交叉熵损失 