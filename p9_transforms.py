from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# python的用法 --> 数据类型 
# transforms.ToTensor 1.如何使用; 2.Tensor数据类型

img_path = 'PyTorch-note/data/hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(img_path)
# print(img)

# 2.Tensor数据类型
writer = SummaryWriter('PyTorch-note/logs')



# 1.如何使用transforms
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img) # img图片转换成tensor的img

# print(tensor_img)

writer.add_image('Tensor_img', tensor_img) # 用tensor形式的图

writer.close()

'''
torchvision 中的 transforms 对图片进行变换
        transforms.py工具箱
                |
                V
        totensor; resize... 
              (工具)
                |
                V
            创建具体的工具 
图片  --> transforms.ToTensor()   --> 结果
                |
                V
             使用工具
               输入：
               输出：
        result = tool(input)

'''