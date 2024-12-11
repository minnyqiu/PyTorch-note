import torch

# 测试分类正确率衡量

outputs = torch.tensor([[0.1, 0.2],
                        [0.3, 0.4]])

print(outputs.argmax(1)) # 1 横向 0 纵向

# 0 纵向 tensor([1, 1])
# 1 横向 tensor([1, 1])

preds = outputs.argmax(1)

targets = torch.tensor([0, 1])

print((preds == targets).sum()) # tensor([False,  True]) / tensor(1) 只有一个是对的


