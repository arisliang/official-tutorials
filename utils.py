from mxnet import ndarray as nd

# 计算精度
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()

def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)

# 优化
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad