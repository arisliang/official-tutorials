# DenseNet：稠密连接的卷积神经网络
# http://zh.gluon.ai/chapter_convolutional-neural-networks/densenet-gluon.html

from mxnet import nd
from mxnet.gluon import nn


def conv_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=3, padding=1)
    )
    return out


class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layers):
            self.net.add(conv_block(growth_rate))

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x

dblk = DenseBlock(2, 10)
dblk.initialize()

x = nd.random.uniform(shape=(4, 3, 8, 8))
dblk(x).shape

# 过渡块（Transition Block）
def transition_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return out

tblk = transition_block(10)
tblk.initialize()

tblk(x).shape

# DenseNet
init_channels = 64
growth_rate = 32
block_layers = [6, 12, 24, 16]
num_classes = 10

def dense_net():
    net = nn.Sequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(init_channels, kernel_size=7,
                      strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        # dense blocks
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                net.add(transition_block(channels // 2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
#             nn.AvgPool2D(pool_size=1),
            nn.GlobalAvgPool2D(),
            # TODO: not global pooling?
            nn.Flatten(),
            nn.Dense(num_classes)
        )
    return net

net = dense_net()
net.initialize()
x = nd.random.uniform(shape=(4,3,299, 299))
y = net(x)
print(net)

# 获取数据并训练
import sys
sys.path.append('..')
import utils
import mxnet as mx
from mxnet import init

batch_size = 64
resize=32
learning_rate = 0.1

train_data, test_data = utils.load_data(batch_size=batch_size, resize=resize)

ctx = utils.try_gpu(1)
net = dense_net()
net.initialize(ctx=ctx, init=init.Xavier())
utils.train(net=net, train_data=train_data, test_data=test_data,ctx=ctx,
            batch_size=batch_size, learning_rate=learning_rate)

