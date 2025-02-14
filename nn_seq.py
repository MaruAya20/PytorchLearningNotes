import torch
from torch import nn

from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        # self.conv1 = Conv2d(3,32,5,1,2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32,32,5,1,2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32,64,5,1,2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten1 = Flatten()
        # self.linear1 = Linear(1024,64)
        # self.linear2 = Linear(64,10)

        # 这里由模型shape可以计算padding值和stride值，计算公式在conv2d官方文档里（其实我觉得可以问deepseek）
        #模型示意图详见根文件夹CIFAR10 Model

        #这里讲sequential:
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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten1(x)
        # x = self.linear1(x)
        # x = self.linear2(x)

        x = self.model1(x)
        #用sequential后就不需要每个都写一次了
        return x

tudui = Tudui()

print(tudui)

input = torch.ones((64,3,32,32))    #创建一个所有数都是1的（NCHW型数据）即64个3通道的32x32数据
output = tudui(input)
print(output.shape)   #这样就能检验网络

writer = SummaryWriter("logs/Seq")
writer.add_graph(tudui,input)
writer.close()