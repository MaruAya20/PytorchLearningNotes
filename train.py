import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Tudui


#①dataset
train_data = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)


train_data_size = len(train_data)
print(f"train_data_size:{train_data_size}")  #50000
test_data_size = len(test_data)
print(f"test_data_size:{test_data_size}") #10000

#②dataloader

train_dataloader = DataLoader(train_data,64)
test_dataloader = DataLoader(test_data,64)

#③搭建神经网络
#10分类网络

# class Tudui(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = Sequential(
#             nn.Conv2d(3,32,5,stride=1,padding=2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32,32,5,stride=1,padding=2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32,64,5,1,2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(1024,64),
#             nn.Linear(64,10)
#         )    #序列化
#
#     def forward(self,x):
#         x = self.model(x)
#         return x

#一般都是写在另一个文件里所以丢到model.py里了


#④创建网络模型
tudui = Tudui()

#⑤损失函数
loss_fn = nn.CrossEntropyLoss()

#⑥优化器
learning_rate = 0.01#方便修改
#learning_rate = 1e-2 #科学计数也可以
optimizer = torch.optim.SGD(tudui.parameters(),learning_rate)   #随机梯度下降


#⑦设置训练网络里的一些参数
#训练次数
total_train_step = 0
#测试次数
total_test_step = 0
#轮数
epoch = 10

#添加tensorboard
writer = SummaryWriter("./logs/train")


for i in range(epoch):
    print(f"--------第 {i+1} 轮训练开始-------")

    #训练步骤开始
    tudui.train()       #这一行效果不明显，对特定的Dropout和BatchNorm等Modules才有用，设置为训练模式
    for data in train_dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)

        #调优：
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #计数：
        total_train_step+=1
        if total_train_step %100 == 0:
            print(f"训练次数: {total_train_step} , Loss: {loss.item()}")       #item()转换成真实数字
            writer.add_scalar("train_loss",loss.item(),total_train_step)

                            #每轮训练跑完可以测试一下，评估模型：
    #测试步骤开始
    tudui.eval()       #这一行效果不明显，对特定的Dropout和BatchNorm等Modules才有用，设置为评估模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy/test_data_size}")
    total_test_step+=1
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)

    #保存模型
    torch.save(tudui.state_dict(),"tudui_{}.pth")
    print("模型已保存")

writer.close()