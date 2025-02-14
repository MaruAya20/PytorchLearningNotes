import torch
import torchvision.transforms
from PIL import Image
from model import Tudui

image_path = "./imgs/plane1.png"
image = Image.open(image_path)
print(image)

image = image.convert("RGB")    #PNG是四通道的，多一个透明度通道，要转换回来

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                             torchvision.transforms.ToTensor()])

image = transforms(image)
print(image.shape)

tudui = Tudui()
model_dict3 = torch.load("tudui_{}_GPU.pth")
tudui.load_state_dict(model_dict3)
print(tudui)
image = torch.reshape(image,(1,3,32,32))
tudui.eval()    #走个流程，还是很重要的。
with torch.no_grad():   #这个也是走流程
    output = tudui(image)
print(output)

print(output.argmax(1)) #dog2对了 plane1也对了