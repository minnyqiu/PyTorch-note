from torch.utils.tensorboard import SummaryWriter

# 创建实例
writer = SummaryWriter('PyTorch-note/p7-8/logs') # 写入事件文件

# writer.add_image()
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
