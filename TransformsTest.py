from idlelib.pyparse import trans

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("Logs")
img = Image.open("dataset/train/ants/0013035.jpg")
print(img)

#ToTensor类的实例化及其过程中的转化特性
# 功能：（数据——>tensor类）
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Design Ant",img_tensor)


#Normalize方法
#对数据Normalize（归一化） 参数（[各通道均值][与前面相对应的各通道标准差]） #补充：通道就是RGB通道等内容，一般用RGB通道[][][]共三个通道
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1,1,4],[5,1,4])  #均值和标准差，这里作假定不纠结
img_norm = trans_norm(img_tensor)
#归一化计算公式: input[channel] = input[channel] - mean[channel] / std[channel]
#上面公式就是 input-0.5 * 0.5 = 2*input-1
#则 input[0,1] 时 result[-1,1]
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

#Resize方法
#缩放图片  参数（H,W）
print(img.size)
trans_resize = transforms.Resize((512,512))
#img (PIL类) -> resize -> img_resize (还是PIL类)
img_resize = trans_resize(img)
print(img_resize)       #这里变成512了
#img_resize(PIL类) -> totensor -> img_resize (tensor类)
img_resize = trans_totensor(img_resize) #直接用以前实例化过的totensor成员即可
writer.add_image("Resized",img_resize)

# Compose - resize - 2
#形成类型变换管道
trans_resize_2 = transforms.Resize(1024)
#小知识，图像一般先调整大小，再调整张量，再标准化。
# Compose的参数是——变化成员的数组即可完成一个整体的变化过程(PIL -> PIL(变化过大小)->tensor(张量))
#所以后者的输入类型必须是前者的输出类型。
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resized",img_resize_2,1)

#RandomCrop 随机裁剪
#参数：传一个就是随机裁出正方形，传元组就是裁（H,W）形
trans_random = transforms.RandomCrop((400,500)) #传元组必须小于原图HW值
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW",img_crop,i)





writer.close()