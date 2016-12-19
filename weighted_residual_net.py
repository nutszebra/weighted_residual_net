import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class Conv_BN_ReLU(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        super(Conv_BN_ReLU, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(out_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)

    def __call__(self, x, train=False):
        return F.relu(self.bn(self.conv(x), test=not train))


class Weight(nutszebra_chainer.Model):

    def __init__(self):
        super(Weight, self).__init__()
        modules = []
        modules += [('weight', L.Convolution2D(1, 1, 1, 1, 0, nobias=True))]
        # register layers
        [self.add_link(*link) for link in modules]

    def weight_initialization(self):
        self.weight.W.data = np.zeros((1, 1, 1, 1), dtype=np.float32)

    def count_parameters(self):
        return 1

    def __call__(self, x):
        batch, channel, height, width = x.data.shape
        h = self.weight(F.reshape(x, (batch, 1, channel * height, width)))
        return F.reshape(h, (batch, channel, height, width))


class Conv_BN_ReLU_Conv_BN_ReLU_Res(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1), probability=1.0):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.probability = probability
        super(Conv_BN_ReLU_Conv_BN_ReLU_Res, self).__init__()
        modules = []
        modules += [('conv1', Conv_BN_ReLU(in_channel, out_channel, filter_size[0], stride[0], pad[0]))]
        modules += [('conv2', Conv_BN_ReLU(out_channel, out_channel, filter_size[1], stride[1], pad[1]))]
        modules += [('weight', Weight())]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return x
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    def maybe_pooling(self, x):
        if 2 in self.stride:
            return F.average_pooling_2d(x, 1, 2, 0)
        return x

    def __call__(self, x, train=False):
        if train is True and self.probability <= np.random.rand():
            # do nothing
            return x
        else:
            batch, channel, height, width = x.data.shape
            _, in_channel, _, _ = self.conv1.conv.W.data.shape
            x = self.concatenate_zero_pad(x, (batch, in_channel, height, width), x.volatile, type(x.data))
            h = self.conv1(x, train)
            h = self.conv2(h, train)
            h = self.weight(h)
            # expectation
            if train is False:
                h = h * self.probability
            h = h + self.concatenate_zero_pad(self.maybe_pooling(x), h.data.shape, h.volatile, type(h.data))
            return h


class ResBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel=3, out_channel=16, n=48, strides=((1, 1),) * 18, probability=(1.0, ) * 18):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n = n
        self.strides = strides
        self.probability = probability
        super(ResBlock, self).__init__()
        modules = []
        for i in six.moves.range(n):
            modules += [('block{}'.format(i), Conv_BN_ReLU_Conv_BN_ReLU_Res(in_channel, out_channel, (3, 3), strides[i], (1, 1), probability[i]))]
            in_channel = out_channel
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=False):
        for i in six.moves.range(self.n):
            x = self['block{}'.format(i)](x, train=train)
        return x


class WeightedResidualNetwork(nutszebra_chainer.Model):

    def __init__(self, category_num, out_channels=(16, 32, 64), N=(198, 198, 198), p=(1.0, 0.5)):
        self.category_num = category_num
        self.out_channels = out_channels
        self.N = N
        super(WeightedResidualNetwork, self).__init__()
        # conv
        modules = [('conv1', Conv_BN_ReLU(3, out_channels[0], 3, 1, 1))]
        # strides
        strides = [[(1, 1) for _ in six.moves.range(n)] for n in N]
        for i in six.moves.range(1, len(out_channels)):
            strides[i][0] = (1, 2)
        # drop path
        drop_probability = WeightedResidualNetwork.linear_schedule(p[0], p[1], N)
        # res block
        in_channel = out_channels[0]
        for i in six.moves.range(len(out_channels)):
            modules += [('res_block{}'.format(i), ResBlock(in_channel, out_channels[i], N[i], strides[i], drop_probability[i]))]
            in_channel = out_channels[i]
        # linear
        modules.append(('linear', Conv_BN_ReLU(in_channel, category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.name = 'residual_network_{}_{}_{}'.format(category_num, out_channels, N)

    @staticmethod
    def linear_schedule(bottom_layer, top_layer, N):
        total_block = sum(N)

        def y(x):
            return (float(-1 * bottom_layer) + top_layer) / (total_block) * x + bottom_layer
        theta = []
        count = 0
        for num in N:
            tmp = []
            for i in six.moves.range(count, count + num):
                tmp.append(y(i))
            theta.append(tmp)
            count += num
        return theta

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=False):
        h = self.conv1(x, train)
        for i in six.moves.range(len(self.out_channels)):
            h = self['res_block{}'.format(i)](h, train)
        batch, channels, height, width = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channels, 1, 1))
        return F.reshape(self.linear(h, train), (batch, self.category_num))

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
