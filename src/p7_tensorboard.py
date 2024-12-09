from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

'''SummaryWriter记录和可视化训练过程'''

# 创建实例
writer = SummaryWriter('PyTorch-note/p7-8/logs') # 写入事件文件

img_path = 'PyTorch-note/data/hymenoptera_data/train/ants/0013035.jpg'
# img_path = 'PyTorch-note/data/hymenoptera_data/train/bees/16838648_415acd9e3f.jpg'

img_PIL = Image.open(img_path)
img_array = np.array(img_PIL) # 转成numpy输入add_image函数
print(type(img_array))
print(img_array.shape) # (512, 768, 3)

writer.add_image('test', img_array, 1, dataformats='HWC') # 用numpy形式的图
# tag 训练图表的title: test
# shape default (H, W, 3); math (H, W, 3) use para <dataformats='HWC'>
# height, width, channel
# y = 2x

for i in range (100): # i从0到99
    
    writer.add_scalar('y=2x', 2*i, i) 
    # tag 训练图表的title
    # scalar_value 需要保存的数值, y轴
    # global_step 训练的步数，x轴

writer.close()

# 打开事件文件
# terminal 
# tensorboard --logdir=PyTorch-note/p7-8/logs
# 指定port防止与其他用户冲突
# tensorboard --logdir=PyTorch-note/p7-8/logs --port=6007




# help(SummaryWriter)

# tensorboard使用
