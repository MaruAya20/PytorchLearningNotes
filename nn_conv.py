import torch
import torch.nn.functional as F
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

input = torch.reshape(input,(1,1,5,5))  #一个数据一个通道（平面）5x5
kernel = torch.reshape(kernel,(1,1,3,3)) #将尺寸修改为4个数
print(input.shape)
print(kernel.shape)

#conv2d 计算卷积核，参数一览：
#前两个分别为输入矩阵，卷积核
# stride参数就是计算完一轮卷积核kernel移动的步长
# padding参数就是判定填充，给上下左右各扩大n圈(扩大的格子内数据默认为0)

output = F.conv2d(input,kernel,stride=1)
print(output)

output2 =F.conv2d(input,kernel,stride=2)
print(output2)

output3 = F.conv2d(input,kernel,stride=1,padding=1)
print(output3)
