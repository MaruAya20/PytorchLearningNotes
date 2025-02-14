import torch
import torchvision
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from torch.fx.experimental.meta_tracer import torch_where_override
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        #这里应该算是把conv1定义了一下怎么卷积化
    def forward(self,x):
        x = self.conv1(x)
        return x

tudui = Tudui()
print(tudui)

writer = SummaryWriter("./logs")

step = 0

for data in dataloader:
    imgs,targets = data
    output = tudui(imgs)  #经过网络后的一个结果
    writer.add_images("input",imgs,step)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step = step+1

#就是写了个骨架，然后导入数据集让他在神经网络里走了一遍，再用dataloader输出到tensorboard，很综合的一次课