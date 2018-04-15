from mxnet import ndarray as nd
import mxnet as mx

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
    except:
        ctx = mx.cpu()
    return ctx
