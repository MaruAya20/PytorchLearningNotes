import torch
from torch import nn
from torch.nn import Sequential


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            nn.Conv2d(3,32,5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )    #序列化

    def forward(self,x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    #测试用的,学过Python不要忘
    tudui = Tudui()
    input = torch.ones(64,3,32,32)
    output= tudui(input)
    print(output.shape)