import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1,2,3],dtype=torch.float32)
targets = torch.tensor([1,2,5],dtype=torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))   #1 batch_size的 1channel 1行 3列
targets = torch.reshape(targets,(1,1,1,3))
#新版对batch_size不做要求所以一般不用reshape

loss = L1Loss(reduction="sum")
result = loss(inputs,targets) #绝对差 2/3
print(result)

loss = MSELoss()
result = loss(inputs,targets)   #均方误差(x-y)^2    2^2 / 3 = 1.333
print(result)


#交叉熵 ，这个用来训练分类问题的。计算公式……
#公式在这：   Loss = -x[class] + ln( Σ exp(x[j]) )  后来改了改什么样……有点看不懂

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss = CrossEntropyLoss()
result = loss(x,y)
print(result)

#loss function很多，实际使用时挑着用，然后一定要注意输入输出的shape