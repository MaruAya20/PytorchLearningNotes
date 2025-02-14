import torch
import torchvision.datasets
from torch import nn

from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model1 = Sequential(Conv2d(3,32,5,1,2),
                                 MaxPool2d(2),
                                 Conv2d(32, 32, 5, 1, 2),
                                 MaxPool2d(2),
                                 Conv2d(32, 64, 5, 1, 2),
                                 MaxPool2d(2),
                                 Flatten(),
                                 Linear(1024,64),
                                 Linear(64,10)
        )
    def forward(self,x):
        x = self.model1(x)
        return x

loss = CrossEntropyLoss()
tudui = Tudui()

for data in dataloader:
    imgs,targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs,targets)
    result_loss.backward()  #计算出了一个梯度，下一节课优化器optimizer，他们利用梯度对网络参数进行更新
    print("ok")
    #grad梯度，反向传播各节点中有更新梯度，在更新输出过程中调整梯度以达到降低loss的程度