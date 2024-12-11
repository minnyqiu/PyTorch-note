import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time

# 准备数据集

train_data = torchvision.datasets.CIFAR10(root='PyTorch-note/data', train=True, transform=torchvision.transforms.ToTensor(),download=True)

test_data = torchvision.datasets.CIFAR10(root='PyTorch-note/data', train=False, transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

# 如果train_data_size = 10, 训练数据长度为10
print(f'train data length:{train_data_size}') # 格式化字符串
print(f'test data length:{test_data_size}')

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

# 创建网络模型
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda() # 可用再转移到cuda

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda() # 转移到cuda

# 优化器
# 1e-2 = 1 * (10)^(-2)
learning_rate = 0.01 # 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的参数

# 记录训练次数
total_train_step = 0

# 记录测试次数
total_test_step = 0

# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter('/home/data/minyu/PyTorch-note/logs/p24')

start_time = time.time() # 记录开始时间

for i in range(epoch):
    print(f'---------- 第{i+1}轮训练开始 ----------')

    # 训练步骤开始 
    tudui.train()  # 作用? BatchNorm module 
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda() # 转移到cuda
        if torch.cuda.is_available():
            targets = targets.cuda() # 转移到cuda
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 用优化器进行梯度清零 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1 # 每次都打印
        # 逢百打印
        if total_train_step % 100 == 0:
            end_time = time.time() # 从训练开始到当前完成 100 次训练步骤所用的时间
            print(f'100次训练耗时:{end_time - start_time:.2f}秒')
            print(f'训练次数: {total_train_step}, Loss: {loss.item()}')
            writer.add_scalar('train_loss', loss.item(), total_train_step) 
            start_time = end_time  # 更新开始时间 计算每100次所用的时间

    # 测试步骤开始
    tudui.eval()  # 作用? BatchNorm module 
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 不要梯度
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda() # 转移到cuda
            if torch.cuda.is_available():
                targets = targets.cuda() # 转移到cuda
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets) # loss是tensor类型
            total_test_loss = total_test_loss + loss.item() # 用.item()变成数字
            # 
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print(f'整体测试集上的loss{total_test_loss}')
    print(f'整体测试集上的正确率{total_accuracy/test_data_size}') # 

    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy/test_data_size, total_test_step)

    total_test_step = total_test_step + 1

    # 保存模型
    torch.save(tudui, f'PyTorch-note/output/tudui_{i}_gpu.pth')
    print('模型已保存')

writer.close()

# Terminal 
# tensorboard --logdir=PyTorch-note/logs/p23


""" 
1.网络模型 tudui

2.数据 (输入, 标注) imgs, targets = data

3.损失函数 loss_fn

对以上三个调用.cuda() 再返回

"""

