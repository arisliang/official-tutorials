from mxnet.gluon import nn
from mxnet import nd
import utils

class _Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(_Residual, self).__init__(**kwargs)
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
            
class ResidualIdentity(_Residual):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(ResidualIdentity, self).__init__(channels, same_shape, **kwargs)
    
    def forward(self, x):
#         print('x.shape:', x.shape)
        
        out = self.conv1(nd.relu(self.bn1(x)))
#         print('out.shape:', out.shape)
        
        out = self.conv2(nd.relu(self.bn2(out)))
#         print('out.shape:', out.shape)
        
        if not self.same_shape:
            x = self.conv3(x)
#             print('x.shape:', x.shape)
            
        return out + x
        
        
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
#         print('same_shape:', self.same_shape)
#         print('x.shape:', x.shape)

        out = self.conv1(nd.relu(self.bn1(x)))
#         print('out.shape:', out.shape)

        out = self.conv2(nd.relu(self.bn2(out)))
#         print('out.shape:', out.shape)

        out = self.conv3(nd.relu(self.bn3(out)))
#         print('out.shape:', out.shape)

        if not self.same_shape:
            x = self.conv4(x)
#             print('x.shape:', x.shape)

        return out + x
    
# 构建ResNet
class _ResNet(nn.Block):
    def __init__(self, version=2, **kwargs):
        super(_ResNet, self).__init__(**kwargs)
        self.version = version
        if version == 1:
            Residual = _Residual
            ResidualBottleneck = _ResidualBottleneck
        elif version == 2:
            Residual = ResidualIdentity
            ResidualBottleneck = ResidualIdentityBottleneck
        
    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i+1, out.shape))
        return out


class ResNet18(_ResNet):
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

    
class ResNet34(_ResNet):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet34, self).__init__(**kwargs)
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
                ResidualIdentity(64),
                ResidualIdentity(64)
            )
            # block 3
            b3 = nn.Sequential()
            b3.add(
                ResidualIdentity(128, same_shape=False),
                ResidualIdentity(128),
                ResidualIdentity(128),
                ResidualIdentity(128)
            )
            # block 4
            b4 = nn.Sequential()
            b4.add(
                ResidualIdentity(256, same_shape=False),
                ResidualIdentity(256),
                ResidualIdentity(256),
                ResidualIdentity(256),
                ResidualIdentity(256),
                ResidualIdentity(256)
            )
            # block 5
            b5 = nn.Sequential()
            b5.add(
                ResidualIdentity(512, same_shape=False),
                ResidualIdentity(512),
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
            
class ResNet50(_ResNet):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet50, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                ResidualIdentityBottleneck(64, 256, same_shape=False),
                ResidualIdentityBottleneck(64, 256),
                ResidualIdentityBottleneck(64, 256)
            )
            # block 3
            b3 = nn.Sequential()
            b3.add(
                ResidualIdentityBottleneck(128, 512, same_shape=False),
                ResidualIdentityBottleneck(128, 512),
                ResidualIdentityBottleneck(128, 512),
                ResidualIdentityBottleneck(128, 512)
            )
            # block 4
            b4 = nn.Sequential()
            b4.add(
                ResidualIdentityBottleneck(256, 1024, same_shape=False),
                ResidualIdentityBottleneck(256, 1024),
                ResidualIdentityBottleneck(256, 1024),
                ResidualIdentityBottleneck(256, 1024),
                ResidualIdentityBottleneck(256, 1024),
                ResidualIdentityBottleneck(256, 1024)
            )
            # block 5
            b5 = nn.Sequential()
            b5.add(
                ResidualIdentityBottleneck(512, 2048, same_shape=False),
                ResidualIdentityBottleneck(512, 2048),
                ResidualIdentityBottleneck(512, 2048)
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

class ResNet101(_ResNet):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet101, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                ResidualIdentityBottleneck(64, 256, same_shape=False),
                ResidualIdentityBottleneck(64, 256),
                ResidualIdentityBottleneck(64, 256)
            )
            # block 3
            b3 = nn.Sequential()
            b3.add(
                ResidualIdentityBottleneck(128, 512, same_shape=False),
                ResidualIdentityBottleneck(128, 512),
                ResidualIdentityBottleneck(128, 512),
                ResidualIdentityBottleneck(128, 512)
            )
            # block 4
            b4 = nn.Sequential()
            b4.add(
                ResidualIdentityBottleneck(256, 1024, same_shape=False)
            )
            for _ in range(22):
                b4.add(ResidualIdentityBottleneck(256, 1024))
            # block 5
            b5 = nn.Sequential()
            b5.add(
                ResidualIdentityBottleneck(512, 2048, same_shape=False),
                ResidualIdentityBottleneck(512, 2048),
                ResidualIdentityBottleneck(512, 2048)
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

class ResNet152(_ResNet):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet152, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                ResidualIdentityBottleneck(64, 256, same_shape=False),
                ResidualIdentityBottleneck(64, 256),
                ResidualIdentityBottleneck(64, 256)
            )
            # block 3
            b3 = nn.Sequential()
            b3.add(
                ResidualIdentityBottleneck(128, 512, same_shape=False),
            )
            for _ in range(7):
                b3.add(ResidualIdentityBottleneck(128, 512))
            # block 4
            b4 = nn.Sequential()
            b4.add(
                ResidualIdentityBottleneck(256, 1024, same_shape=False)
            )
            for _ in range(35):
                b4.add(ResidualIdentityBottleneck(256, 1024))
            # block 5
            b5 = nn.Sequential()
            b5.add(
                ResidualIdentityBottleneck(512, 2048, same_shape=False),
                ResidualIdentityBottleneck(512, 2048),
                ResidualIdentityBottleneck(512, 2048)
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
