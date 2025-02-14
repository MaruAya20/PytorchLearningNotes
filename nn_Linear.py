import torch
import torchvision
from torch.nn import Linear
from torch.utils.data import  DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608,10)

    def forward(self,input):
        output = self.linear1(input)
        return output

tudui = Tudui()


for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    output = torch.reshape(imgs,(1,1,1,-1)) #-1项可以让程序自行计算修改后的shape
    #torch.flatten() 将多维展开层一维的数据
    torch.flatten(output)
    print(output.shape)
    output = tudui(output)
    print(output.shape)