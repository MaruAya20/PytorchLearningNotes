
#Pytorch快速入门学习笔记
#没错这里就是记事本

#这里是跟个风学个pytorch，暑假没什么事干不太对。
#其实这些py应该放src里才对，手滑。

#anaconda的prompt里创建环境指令：conda create -n (名字) python=3.6(这个是python版本参数)

#package(一个工具箱，也就是环境)
#多个分割区组成，分割区里放工具
#dir() 打开操作
#help()说明书

#dir(pytorch)  输出分割区的名字

#dir(pytorch.3) 打开输出分割区“3” 输出里面的工具。

#help(pytorch.3.a) 输出工具的功能。

#上面函数方便我们学习python包的工具

#ps：有点太照顾不学语言却想学pytorch的人了



#  Dataset 类
# 提供方式获取数据以及label


#照片的数据读取整理已经写在read_data01了，用了新类
#重点是把数据与标签label对应，除了给文件直接命名label以外，还有一种形式
#就是另外开一个文件夹里面存放txt文件以保存一个数据包含的大量label


#Tensorboard 的使用
#Tensorboard详见test_tb.py        #这个用多就记得了


#图像变换transform的使用
#transform详见TransformsTest.py
#省流：transform类内的多个成员及其构造方式：
# Totensor Normalize Resize Compose RandomCrop 等实例化（带参）的使用
#总结：①关注输入输出类型别弄错了②多看官方文档（准确）③关注参数Args都有什么


#dataloader 类   //dataset准备牌，dataloader发牌
#为后面网络提供数据形式
#省流：Dataloader参数：dataset,batch_size,shuffle,num_worker=0,drop_last=True 的各个意思

#Neuro Network
#构建神经网络的骨架，详见nn_module.py
#省流：写一个model类，完成他的初始化重写以及forward函数的实现。
#简单讲了一下conv2d（二维卷积计算）
#参数一览：input,kernel,stride=1,padding=1 理解一下

#卷积层（Convolution layers）
#这里基本没啥不一样也是讲conv2d，但是kernel不需要给出，一般是学习学出来的只要定义kernel_size即可
#还有一点就是in_channels和out_channels，一般都为1，如果out_channels为2，机器会对输入的矩阵用两
# 个或许不同的卷积核来进行计算，输出两个矩阵
#省流：在骨架中重写conv1(用到conv2d)，再导入数据集让他在神经网络里走一圈，最后用dataloader输出到Tensorboard上。

#空洞卷积->设置dilation使得卷积核各元素间有距离,变成有洞的形式

#最大池化（下采样，Pooling layers）
#最大池平均池……同上最重要是MaxPool2d
#复习一下ceil和floor,ceil向上取整,floor向下取整
#跟卷积找的方式一样,而且更简单,直接找池化核内最大数作为输出
#重点是,strides默认值为kernel_size,如果步长大超出了,就要看ceil_mode,True就会保留最后剩余的数据，False就不会
#默认情况ceil_mode是False

#非线性激活（Non-linear Activation）
#就是按给一个输入值，非线性函数输出值，常见的有ReLU、Sigmoid
#详见nn_ReLU.py



#其他层的简略介绍

#正则化层（Normalization Layers）
#用的不多，BatchNorm2d(Channel数，其他参数默认即可)，使用正则化层可以加快神经网络的训练。
#并没有实例操作

#循环层（Recurrent Layers）
#一样的不常用，搞循环神经网络的

#Transformer Layers
#略

#线性层（Linear Layers）
#这个后面会讲

#（Dropout Layers）
#随机失真，防止过拟合（将部分张量随机取零，概率为：n）

#Sparse Layers
#自然语言处理有用

#Distance Functions、Loss Functions……后面再说吧


#线性层（Linear Layers）
#执行 Input Layers -> Hidden Layers -> Output Layers 的过程
#Hidden层的计算: K1*x1+b1 + K2*x2+b2 + .... + Kd*xd+bd = g1
#k是权重，b是偏置
#至于什么用法，可以看一下vgg16模型示意图（就在根文件夹）最后的softmax就是通过线性层
#由1x1x4096的input，经过hidden layers，变成1x1x1000的output

#神经网络基础结构大概那么多


#squential，类似compose，可以整合卷积conv2d，relu，等操作
#详见nn_seq.py，顺便CIFAR10 Model也学习怎么实现了（三卷积三最大池化+平铺+线性化）可以去看看


#损失函数(Loss Function)
#为了让实际输出和target目标尽量相符，Loss要尽量小
#Loss Function为我们更新输出提供一个有效的依据（反向传播）
#L1Loss： 求差距和再取平均。
#再次提醒，要学会任何成员，你必须要知道Shape，也就是输入输出的规格，翻官方文档要好好看这些内容

#优化器(optimizer)
#详见nn_optim.py
#原理是根据grad值进行参数调整，以达到降低loss值的目的
#过程：Loss -> Grad -> Optim  -> 降低后的Loss -> Grad....


#网络模型的使用和修改
#详见model_pretrained.py
#网络模型你可以看成一个类，里面他的各种module就是成员，你可以按常见的访问方法来修改或者添加他们。

#网络模型的存储和读取。
#详见model_save.py 以及 model_load.py
#网络模型存储两个内容：模型自身以及网络参数，可以单独存储网络参数成字典文件，也可以直接全部存储单独文件。


#完整的模型训练套路：（仍然是CIFAR10的分类问题）
#详见train.py

#利用GPU训练
#详见train_GPU_1.py，原理是将网络模型，训练或测试中的数据，以及损失函数调用.cuda()，让GPU计算这些数据。
#如果条件不好可以去google的colab上借用他们的gpu跑，效果还行，上述py里的训练时间大概在0.9秒/100次
#.to(device)方法  torch.device("cuda")方法

#完整的模型测试套路（demo）
# #利用训练好的模型给它提供输入，详见test.py


#完结撒花~
#pytorch入门内容学习完了，往后可以通过查看一些开源项目来进行更深层次的学习，构建自己的项目也是可行的。



#相关网站：

#【PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】】
# https://www.bilibili.com/video/BV1hE411t7RN/?p=26&share_source=copy_web&vd_source=75c737e9e1d2ad9cf656e6e37bc0a7b5
#https://github.com/MaruAya20  建个github玩玩，把学习笔记丢进去当填充物哈哈

#2025.02.14