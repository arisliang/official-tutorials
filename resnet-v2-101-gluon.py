# ResNet：深度残差网络
# http://zh.gluon.ai/chapter_convolutional-neural-networks/resnet-gluon.html

from mxnet.gluon import nn
from mxnet import nd
import utils
from resnets import *

# 输入输出通道相同
blk = ResidualIdentity(3)
blk.initialize()

x = nd.random.uniform(shape=(4,3,96,96))
y = blk(x)
print('y.shape:', y.shape)
print(blk)

# 输入输出通道不同
blk2 = ResidualIdentity(8, same_shape=False)
blk2.initialize()
# print(blk2)
y2 = blk2(x)
print('y2.shape:',y2.shape)
print(blk2)

net = ResNet101(10, verbose=True)
net.initialize()

x = nd.random.uniform(shape=(4,3,299, 299))
y = net(x)
print(net)

# 读取数据
import sys
sys.path.append('..')
import utils
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet import image
from mxnet import init

batch_size = 128
resize=299
learning_rate=0.01

train_data, test_data = utils.load_data(batch_size=batch_size, resize=resize)

ctx = utils.try_gpu(0)
net = ResNet101(10, verbose=False)
net.initialize(ctx=ctx, init=init.Xavier())

utils.train(net=net, train_data=train_data, test_data=test_data,ctx=ctx,
            batch_size=batch_size, learning_rate=learning_rate)