import torch
import torchvision.models
import model_save
from pycparser.c_ast import Default

#2.6版本后，torch为了防止恶意构造pickle数据，把weights_only设置成True（大概就是安全模式加载的意思）
#读取方式1（结构和参数）
model_1 = torch.load("vgg16_method1.pth",weights_only=False)
print(model_1)

#读取方式2（读取参数字典）
model_dict2 = torch.load("vgg16_method2.pth")
print(model_dict2)  #喏，字典。

vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(model_dict2)

#陷阱1
model = torch.load('tudui_method1.pth',weights_only=False)
print(model)
#其实不是陷阱，你要导入这个模型的定义python才看得懂这怎么load
