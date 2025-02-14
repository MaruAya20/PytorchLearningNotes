import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from urllib3.filepost import writer

#准备测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform= torchvision.transforms.ToTensor())

#windows下num_workers不能设置大于0
test_loader = DataLoader(test_data,64,True,num_workers=0,drop_last=True)
#参数一览：
#第一个dataset
#batch_size: 每一个小包内的数据份数
#shuffle: 洗牌（随机）
#num_workers: 处理的进程个数(windows不能大于0，无脑填0)
#drop_last: 丢弃最后一个份数不完整的数据包

#测试数据集第一张图片及target
img,target = test_data[0]
# print(img.shape)    #3通道，32x32，的图片
# print(target)       #3

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    #从loader里取回打包好的图片集和target集
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)     #[4,3,32,32] 4张3通道的32x32图片
        # print(targets)        #[1,0,9,0]   随机的，
        writer.add_images("Epoch:{}".format(epoch),imgs,step)       #这里如果shuffle:True才会使第二轮与第一轮不同
        step = step+1

writer.close()