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
optim = torch.optim.SGD(tudui.parameters(),lr=0.01) #传入神经网络参数以及训练速度

for epoch in range(20):     #进行多轮的优化

    for data in dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs,targets)
        optim.zero_grad()       # 清零上个循环记录的梯度
        result_loss.backward()  # 计算出了一个梯度，优化器optimizer，他们利用梯度对网络参数进行更新
        optim.step()   #利用grad对参数更新
        #grad梯度，反向传播各节点中有更新梯度，在更新输出过程中调整梯度以达到降低loss的程度