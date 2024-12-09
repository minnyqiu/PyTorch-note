import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])

# 下载训练测试数据集, ToTensor应用到数据集中每一张
train_set = torchvision.datasets.CIFAR10(
    root = 'PyTorch-note/data', train = True, transform = dataset_transform, download = True)
test_set = torchvision.datasets.CIFAR10(
    root = 'PyTorch-note/data', train = False, transform = dataset_transform, download = True)

# print(test_set[0]) # 查看测试集中的第一个
# print(test_set.classes) # 查看测试集中类别

# img, target = test_set[0]
# print('img:', img)
# print('target:', target)
# print(test_set.classes[target])
# # img.show() 

# print(test_set[0])

writer = SummaryWriter('PyTorch-note/logs/p10')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()
