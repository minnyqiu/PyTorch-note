from torch.utils.data import Dataset
from PIL import Image
import os

# img_path = 'PyTorch-note/p5/hymenoptera_data/train/ants/0013035.jpg'

class Mydata(Dataset):

    def __init__ (self, root_dir, label_dir):
        self.root_dir = root_dir # 根目录 训练文件夹
        self.label_dir = label_dir # 标签目录 训练文件夹里的文件名
        self.path = os.path.join(self.root_dir, self.label_dir) # 拼接根目录和标签
        self.img_path = os.listdir(self.path) # 所有文件和文件夹的名称列表
    
    def __getitem__ (self, idx):
        img_name = self.img_path[idx] # 图像路径在列表中的位置
        img_item_path = os.path.join(self.root_dir, self.label_dir,img_name) # 每张图的路径
        img = Image.open(img_item_path) # 每张图
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)


root_dir = 'PyTorch-note/p5/hymenoptera_data/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'

ants_dataset = Mydata(root_dir, ants_label_dir)
bees_dataset = Mydata(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset

# Dataset 提供一种方式获取数据及其label
# - 如何获取每一个数据及其label
# - 总共有多少数据
# dataloader 为网络提供不同的数据形式