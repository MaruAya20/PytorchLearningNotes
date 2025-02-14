import torch
import torchvision
from torch import nn

#模型存储与读取

vgg16 = torchvision.models.vgg16(weights=None) #不用我再说明pretrained更名为weights这回事吧。
#保存方式1（模型结构+参数）
torch.save(vgg16,"vgg16_method1.pth")

#保存方式2（只把参数保存成字典）（官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

# 陷阱？
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3)

    def forward(self,x):
        x =  self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui,"tudui_method1.pth")