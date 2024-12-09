from csv import writer
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备测试数据集
test_set = torchvision.datasets.CIFAR10(r'PyTorch-note/data',download=True, train = False, 
                                        transform=torchvision.transforms.ToTensor())

# batch_size=4每次从测试集中取四个数据test_set[0]~test_set[3]的img和target进行打包
# drop_last=False/True 是否保留最后一个batch_size
# shuffle=True/False 每一个Epoch图片选取是否打乱, False时, Epoch0和Epoch1相同
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

# 测试集中第一张图片shape及target
img, target = test_set[0]

print(img.shape)

print(target)

# 示意图

# dataset 数据集定义 # getitem() return img, target

# DataLoader(batch_size=4)

# img0, target0 = test_set[0]
# img1, target1 = test_set[1]
# img2, target2 = test_set[2]
# img3, target3 = test_set[3]

# img叠在一起打包imgs, target叠在一起打包targets

# imgs, targets 作为DataLoader中的返回

# 用SummaryWriter展示
writer = SummaryWriter('/home/data/minyu/PyTorch-note/logs/p11')
# shuffle=True/False
for epoch in range(2):

    step = 0

    # 取出每一个返回
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images('Epoch: {}'.format(epoch), imgs, step)
        step = step + 1

writer.close()
 