import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

writer = SummaryWriter("logs")      #运行后生成事件文件

# writer.add_image()  *** ctrl+/ 注释本行代码
for i in range(100):
    writer.add_scalar("y=x",i,i)   #scalar标量



#Terminal运行以开启tensorboard --logdir=logs --port=6007  参数什么意思你应该也清楚不说了
#这个tensorboard可以在网络上打开，还挺有意思，他会给你开一个本地窗口。

image_path = "dataset/train/ants/2278278459_6b99605e50.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("designant",img_array,1,dataformats='HWC') #最后一个是确认参数类型的重写
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

writer.close()
