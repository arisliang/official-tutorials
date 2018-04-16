# ResNet：深度残差网络
# http://zh.gluon.ai/chapter_convolutional-neural-networks/resnet-gluon.html

from mxnet.gluon import nn
from mxnet import nd
from utils import *


class _Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(_Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                               strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1,
                               strides=strides)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                   strides=strides)


class ResidualIdentity(_Residual):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(ResidualIdentity, self).__init__(channels, same_shape, **kwargs)

    def forward(self, x):
        #         print('x.shape:', x.shape)

        out = self.conv1(nd.relu(self.bn1(x)))
        #         print('out.shape:', out.shape)

        out = self.conv2(nd.relu(self.bn2(x)))
        #         print('out.shape:', out.shape)

        if not self.same_shape:
            x = self.conv3(x)
        #             print('x.shape:', x.shape)

        return out + x

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


class _ResidualBottleneck(nn.Block):
    def __init__(self, channels_in, channels_out, same_shape=True, **kwargs):
        super(_ResidualBottleneck, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels_in, kernel_size=1,
                               strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels_in, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(channels_out, kernel_size=1,
                               strides=1)
        self.bn3 = nn.BatchNorm()
        if not same_shape:
            self.conv4 = nn.Conv2D(channels_out, kernel_size=1,
                                   strides=strides)


#     def forward(self, x):
# #         print('same in / out shape:', self.same_shape)
# #         print('x.shape:', x.shape)
#         out = nd.relu(self.bn1(self.conv1(x)))
# #         print(out.shape)
#         out = nd.relu(self.bn2(self.conv2(out)))
# #         print(out.shape)
#         out = self.bn3(self.conv3(out))
# #         print(out.shape)
#         if not self.same_shape:
#             x = self.conv4(x)
# #             print('x.shape:',x.shape)
#         return nd.relu(out + x)
class ResidualIdentityBottleneck(_ResidualBottleneck):
    def __init__(self, channels_in, channels_out, same_shape=True, **kwargs):
        super(ResidualIdentityBottleneck, self).__init__(channels_in, channels_out, same_shape, **kwargs)

        def forward(self, x):
            #             print('x.shape:', x.shape)

            out = self.conv1(nd.relu(self.bn1(x)))
            #             print('out.shape:', out.shape)

            out = self.conv2(nd.relu(self.bn2(x)))
            #             print('out.shape:', out.shape)

            out = self.conv3(nd.relu(self.bn3(x)))
            #             print('out.shape:', out.shape)

            if not self.same_shape:
                x = self.conv3(x)
            #                 print('x.shape:', x.shape)

            return out + x


# 构建ResNet
class ResNet18(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet18, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                ResidualIdentity(64),
                ResidualIdentity(64)
            )
            # block 3
            b3 = nn.Sequential()
            b3.add(
                ResidualIdentity(128, same_shape=False),
                ResidualIdentity(128)
            )
            # block 4
            b4 = nn.Sequential()
            b4.add(
                ResidualIdentity(256, same_shape=False),
                ResidualIdentity(256)
            )
            # block 5
            b5 = nn.Sequential()
            b5.add(
                ResidualIdentity(512, same_shape=False),
                ResidualIdentity(512)
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

model = ResNet18(10, verbose=True)
model.initialize()

x = nd.random.uniform(shape=(4,3,96, 96))
y = model(x)
print(model)

batch_size = 128
train_data, test_data = utils.load_data(batch_size, resize=224)

ctx = utils.try_gpu(0)
model = ResNet18(10, verbose=False)
model.initialize(ctx=ctx, init=init.Xavier())

# 训练
import time
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainable_params = model.collect_params()
print(trainable_params)

trainer = gluon.Trainer(trainable_params, 'sgd', {
    'learning_rate': 0.05
})

for epoch in range(10):
    time_start = time.time()
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = model(data.as_in_context(ctx))
            loss = softmax_cross_entropy(output, label)

        # b1_params = model.net[0]
        # print(b1_params)

        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_data, model, ctx)
    print("Epoch %d. Loss: %.4f, Train acc %.4f, Test acc %.4f, Time %.0f sec" % (
        epoch, train_loss / len(train_data),
        train_acc / len(train_data), test_acc, time.time() - time_start))