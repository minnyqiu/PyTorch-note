import torchvision

train_set = torchvision.datasets.CIFAR10(root = 'PyTorch-note/data', train = True, download = True)
test_set = torchvision.datasets.CIFAR10(root = 'PyTorch-note/data', train = False, download = True)



