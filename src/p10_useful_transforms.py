from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('PyTorch-note/logs')

img = Image.open('PyTorch-note/data/hymenoptera_data/train/ants/0013035.jpg')
print(img)

# ToTensor使用
trans_totensor = transforms.ToTensor() # 具体的工具
img_tensor = trans_totensor(img) # 使用工具
writer.add_image('ToTensor', img_tensor, 0) # title, tensor

# Normalize使用 <输入为tensor image>
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('normalize', img_norm, 1)

# Resize使用 <输入为PIL>
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image('resize', img_resize, 0)
print(img_resize)

# Compose - Resize - 2
trans_resize_2 = transforms.Resize(125) # 短边缩放至512 长宽比不变 糊了
# PIL -> PIL -> tensor # trans_resize_2, 
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image('resize', img_resize_2, 1)
# print(img_resize)

# RandomCrop
trans_random = transforms.RandomCrop((10,100)) # 125
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop', img_crop, i)
writer.close()

'''
常见的transforms 
1.关注输入和输出类型 看官方文档 看类里面 __init__ 需要什么参数

2.不知道返回值 使用print(), print(type())或者断点debug

输入    *PIL        *Image.open()
输出    *tensor     *ToTensor()
作用    *narrays    *cv.imread()
output[channel] = (input[channel] - mean[channel]) / std[channel]
(input-0.5)/0.5 = 2*input - 1
input[0,1]
result[-1,1]

使用console 关注输出
tensorboard --logdir=PyTorch-note/logs --port=6007
'''