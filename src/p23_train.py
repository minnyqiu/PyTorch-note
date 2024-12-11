from turtle import forward
from scipy import optimize
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from p23_model import *
from torch.utils.tensorboard import SummaryWriter

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

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

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
writer = SummaryWriter('/home/data/minyu/PyTorch-note/logs/p23')

for i in range(epoch):
    print(f'---------- 第{i+1}轮训练开始 ----------')

    # 训练步骤开始 
    tudui.train()  # 作用? BatchNorm module 
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 用优化器进行梯度清零 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1 # 每次都打印
        # 逢百打印
        if total_train_step % 100 == 0:
            print(f'训练次数: {total_train_step}, Loss: {loss.item()}')
            writer.add_scalar('train_loss', loss.item(), total_train_step) 

    # 测试步骤开始
    tudui.eval()  # 作用? BatchNorm module 
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 不要梯度
        for data in test_dataloader:
            imgs, targets = data
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
    torch.save(tudui, f'PyTorch-note/output/tudui_{i}.pth')
    print('模型已保存')

writer.close()

# Terminal 
# tensorboard --logdir=PyTorch-note/logs/p23


""" 
用正确率衡量分类问题

2 x input 
Model (2分类)

输出的形式
outputs = 
[0.1, 0.2], [第0类的概率 第1类的概率]
[0.3, 0.4] 

@ 转换成targets
0       1

用argmax求出横向最大值所在的位置

preds = 
[1]
[1]

inputs target = 
[0]
[1]   

preds == inputs target 判断预测结果
[false, true].sum() = 1 # 求出正确位置的个数

"""
