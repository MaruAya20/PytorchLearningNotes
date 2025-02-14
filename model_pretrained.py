from os.path import split

import torchvision.datasets
from conda.exports import download
from pycparser.c_ast import Default
from sympy.solvers.diophantine.diophantine import Linear
from torch import nn

#写两个点就是返回上级目录
#train_data = torchvision.datasets.ImageNet("./dataset",split='train',download=False,transform=torchvision.transforms.ToTensor())
#这个数据集太大了，147G，不方便，还是不用这个了



#vgg16_false = torchvision.models.vgg16(pretrained=False) #只是加载一个模型
#vgg16_true = torchvision.models.vgg16(pretrained=True)  #会把训练好的参数载入

#因为教程的时效性，好像这里的pretrained参数名已经替换成weights了，并且有可能以后被移除。
#我猜是为了能导入自定义的权重，让模型更灵活吧。
vgg16_true = torchvision.models.vgg16(weights=Default)
vgg16_false = torchvision.models.vgg16(weights=None)
#故改成以上形式

train_data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor,
                                          download=True)

#vgg16可以分出1000个类，但是我们只是想对CIFAR10分类，就需要我们会对现有模型进行改动。


#可以看到最后一层线性层，将4096组分为1000组，那为了CIFAR10，我们可以加一层1000->10的线性层（CIFAR10就10类物品图片）


                                                             #①怎么填加？  内置add_module方法

#vgg16_true.add_module("add_linear",nn.Linear(1000,10))  #就加好了

#print(vgg16_true)       #可以看到在最后多了一个linear

                                                    #②特定位置增加怎么加？  按着位置访问后再调用add_module即可

vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))

print(vgg16_true)       #可以看到加到了classifier

                                                    #③怎么修改？   直接访问赋值即可
vgg16_false.classifier[6] = nn.Linear(4096,10)

print(vgg16_false)      #类似成员数组的访问方法


