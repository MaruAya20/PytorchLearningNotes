import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nn_conv2d import dataloader, writer
from nn_module import output

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,download=True,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,64)

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32) #一般都要是浮点型的tensor数据类型

input = torch.reshape(input,(-1,1,5,5))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__() #对父类进行初始化，补全
        self.maxpool1 = MaxPool2d(3,ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output


tudui = Tudui()
writer = SummaryWriter("logs/MaxPool")
step = 0
for data in dataloader:
    imgs , targets = data
    writer.add_images("input",imgs,step)
    output = tudui(imgs)
    writer.add_images("output",output,step) #这里没有变成多个通道，所以不用reshape

    step = step + 1



writer.close()
#作用：保留数据特征，并保持数据量尽量减小