import mxnet as mx

import sys

sys.path.append('..')
import utils
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet import image
from mxnet import init

# 计算精度
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()

def evaluate_accuracy(data_iterator, net, ctx):
    acc = 0.
    for data, label in data_iterator:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)

# 优化
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def try_gpu(gpu_id=0):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(gpu_id)
        _ = nd.array([0], ctx=ctx)
    except Exception as err:
        print(err)
        ctx = mx.cpu()
    return ctx

def load_data(batch_size, resize=None):
    # 读取数据
    # batch_size = 128
    # resize = 224

    def transform(data, label):
        #     print(data.shape)   # (28, 28, 1)
        #     print(label.shape)    # (1,)
        # change data from batch x height x width x channel
        # to batch x channel x height x width
        #     return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')

        if resize is not None:
            data = image.imresize(data, resize, resize)
        #     data = new_data

        return nd.transpose(data.astype('float32'), (2, 0, 1)) / 255, label.astype('float32')

    #     return data.astype('float32') / 255, label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

    return train_data, test_data


def train(net, train_data, test_data, ctx, batch_size, learning_rate,
          num_epochs=10):
    # 训练
    import time
    from mxnet import gluon

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    trainable_params = net.collect_params()
#     print(trainable_params)

    trainer = gluon.Trainer(trainable_params, 'sgd', {
        'learning_rate': learning_rate
    })

    for epoch in range(num_epochs):
        time_start = time.time()
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        test_acc = utils.evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %.4f, Train acc %.4f, Test acc %.4f, Time %.0f sec" % (
            epoch, train_loss / len(train_data),
            train_acc / len(train_data), test_acc, time.time() - time_start))