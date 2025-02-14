from PIL import Image
from torch.onnx.symbolic_opset9 import tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from test_tb import writer

# tensor类型初识

img_path = "dataset/train/ants/7759525_1363d24e88.jpg"

img = Image.open(img_path)
print(img)
tensor_trans = transforms.ToTensor()  #转换后返回tensor类型
tensor_img = tensor_trans(img)

print(tensor_img)

#为什么需要用到tensor数据类型

#张量  包装了神经网络的绝大多基本量 是神经网络结构的学习基础

writer = SummaryWriter("Logs")

tensor_trans = transforms.ToTensor() #实例化
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img",tensor_img)

writer.close()

#常用的transform过程：  PIL -> Tensor -> Ndarrays
