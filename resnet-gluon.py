# ResNet：深度残差网络
# http://zh.gluon.ai/chapter_convolutional-neural-networks/resnet-gluon.html

from mxnet.gluon import nn
from mxnet import nd
from mxnet import autograd
import utils
from mxnet import init


class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                               strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                   strides=strides)

    def forward(self, x):
        #         print('same in / out shape:', self.same_shape)
        #         print('x.shape:', x.shape)
        out = nd.relu(self.bn1(self.conv1(x)))
        #         print(out.shape)
        out = self.bn2(self.conv2(out))
        #         print(out.shape)
        if not self.same_shape:
            x = self.conv3(x)
        #             print('x.shape:',x.shape)
        return nd.relu(out + x)


# 输入输出通道相同
blk = Residual(3)
blk.initialize()

x = nd.random.uniform(shape=(4, 3, 96, 96))
y = blk(x)
print(y.shape)
print(blk)

# 输入输出通道不同
blk2 = Residual(8, same_shape=False)
blk2.initialize()
y2 = blk2(x)
print(y2.shape)
print(blk2)


# 构建ResNet
class ResNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                Residual(64),
                Residual(64)
            )
            # block 3
            b3 = nn.Sequential()
            b3.add(
                Residual(128, same_shape=False),
                Residual(128)
            )
            # block 4
            b4 = nn.Sequential()
            b4.add(
                Residual(256, same_shape=False),
                Residual(256)
            )
            # block 5
            b5 = nn.Sequential()
            b5.add(
                Residual(512, same_shape=False),
                Residual(512)
            )
            # block 6
            b6 = nn.Sequential()
            b6.add(
                nn.AvgPool2D(pool_size=3),
                nn.Dense(num_classes)
            )
            # chain all blocks together
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out


net = ResNet(10, verbose=True)
net.initialize()

x = nd.random.uniform(shape=(4, 3, 96, 96))
with autograd.record():
    y = net(x)
    print(net)
y.backward()

# 读取数据
batch_size = 128
resize = 224
train_data, test_data = utils.load_data(batch_size=batch_size, resize=resize)

ctx = utils.try_gpu(1)
net = ResNet(10, verbose=False)
net.initialize(ctx=ctx, init=init.Xavier())

utils.train(net, train_data, test_data, ctx, batch_size)
