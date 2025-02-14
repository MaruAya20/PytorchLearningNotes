from modulefinder import Module

import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nn_conv2d import dataloader, writer

input = torch.tensor([[1,-0.5],
                      [-1,3]])

input = torch.reshape(input,(-1,1,2,2))

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,download=True,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU() #这里的inplace是指要不要替换输入变量的值，默认值False
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output


tudui = Tudui()
step = 0
writer = SummaryWriter(".logs/ReLU",)
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,global_step=step)
    output = tudui(imgs)
    writer.add_images("output",output,global_step=step)
    step +=1

writer.close()
#print(output)   #ReLU函数大于0的不变，小于零的变为0