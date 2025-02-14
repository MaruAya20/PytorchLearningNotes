import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

#Ctrl+P 观察参数
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)
#
# img ,target = test_set[0]
# print(img)
# print(target)   #标识
# print(test_set.classes[target])
# img.show()

#CIFAR-10数据集是一个……包含60000个32x32的 十个类型的彩照
#print(test_set[0])  #在构建时传入了变换参数，使得test_set实例变成了tensor型

writer = SummaryWriter("logs/p10")
for i in range(0,10): #0~9
    img,target = test_set[i]
    writer.add_image("test_set",img,i)




writer.close()

