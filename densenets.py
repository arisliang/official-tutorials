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


def dense_net(n_classes, blk_layers, i_channels=64, g_rate=32):
    net = nn.Sequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(i_channels, kernel_size=7,
                      strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        # dense blocks
        channels = i_channels
        for i, layers in enumerate(blk_layers):
            net.add(DenseBlock(layers, g_rate))
            channels += layers * g_rate
            if i != len(blk_layers) - 1:
                net.add(transition_block(channels // 2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
#             nn.AvgPool2D(pool_size=1),
            nn.GlobalAvgPool2D(),
            # TODO: not global pooling?
            nn.Flatten(),
            nn.Dense(n_classes)
        )
    return net


# DenseNet
# block_layers = [6, 12, 24, 16]
# smaller densenet for fashion mnist
block_layers = [e // 2 for e in [6, 12, 24, 16]]
num_classes = 10

net = dense_net(num_classes, block_layers)
net.initialize()
x = nd.random.uniform(shape=(4,3,299, 299))
y = net(x)
print(net)