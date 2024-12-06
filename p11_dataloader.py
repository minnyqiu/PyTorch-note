import torchvision
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10('PyTorch-note/data/cifar-10-batches-py',train = False, 
                                        transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
