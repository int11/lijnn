import lijnn
from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *

import cv2 as cv
import numpy as np
import os


class LeNet_5(Model):
    """
    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    https://ieeexplore.ieee.org/abstract/document/726791
    "Gradient-based learning applied to document recognition"
    1998, Yann LeCun Léon Bottou Yoshua Bengio Patrick Haffner
    params_size = 3,317,546

    frist typical CNN model
    input (32,32)
    real predict shape (28,28)
    it same current 4 padding algorithm
    use Tanh activation function
    use average Pooling
    """

    def __init__(self):
        super().__init__()
        self.conv1 = L.Conv2d(6, 5)
        self.conv2 = L.Conv2d(16, 5)
        self.conv3 = L.Conv2d(120, 5)
        self.fc4 = L.Linear(84)
        self.fc5 = L.Linear(10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.max_pooling(x, kernel_size=2)
        x = F.tanh(self.conv2(x))
        x = F.max_pooling(x, kernel_size=2)
        x = F.tanh(self.conv3(x))

        x = F.reshape(x, (x.shape[0], -1))
        x = F.tanh(self.fc4(x))
        x = self.fc5(x)
        return x


class AlexNet(Model):
    """
    https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    ImageNet Classification with Deep Convolutional Neural Networks
    2012, Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    params_size = 76,009,832

    use Relu activation function - Shoter Training Time
    use 2 Gpu

    use Max Pooling, Dropout, Local Response Normalization

    """

    def __init__(self, output_channel=1000):
        super().__init__()
        self.conv1 = L.Conv2d(96, kernel_size=11, stride=4, pad=0)
        self.conv2 = L.Conv2d(256, kernel_size=5, stride=1, pad=2)
        self.conv3 = L.Conv2d(384, kernel_size=3, stride=1, pad=1)
        self.conv4 = L.Conv2d(384, kernel_size=3, stride=1, pad=1)
        self.conv5 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(output_channel)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.local_response_normalization(x)
        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.relu(self.conv2(x))
        x = F.local_response_normalization(x)
        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x


class ZFNet(Model):
    """
    https://arxiv.org/abs/1311.2901
    Visualizing and Understanding Convolutional Networks
    2013.11.12, Matthew D Zeiler, Rob Fergus
    params_size = 386,539,432

    AlexNet -> ZFNet
    CONV1 : (11, 11) Kernel size, 4 strid -> (7, 7) Kernel size, 2 strid
    CONV3,4,5 : 384, 384, 256  out channels -> 512, 1024, 512 out channels
    """

    def __init__(self, output_channel=1000):
        super().__init__()
        self.conv1 = L.Conv2d(96, kernel_size=7, stride=2, pad=0)
        self.conv2 = L.Conv2d(256, kernel_size=5, stride=1, pad=2)
        self.conv3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4 = L.Conv2d(1024, kernel_size=3, stride=1, pad=1)
        self.conv5 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(output_channel)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.local_response_normalization(x)
        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.relu(self.conv2(x))
        x = F.local_response_normalization(x)
        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x


class VGG16(Model):
    """
    https://arxiv.org/abs/1409.1556
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    2014.9.4, Karen Simonyan, Andrew Zisserman
    params_size = 138,357,544

    """
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz'

    def __init__(self, pretrained=False, output_channel=1000):
        super().__init__()
        self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(output_channel)

        if pretrained:
            weights_path = utils.get_file(VGG16.WEIGHTS_PATH)
            self.load_weights_epoch(weights_path)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        if size:
            image = cv.resize(image, size)
        image = image.astype(dtype)
        image -= np.array([103.939, 116.779, 123.68])
        image = image.transpose((2, 0, 1))
        return image


class GoogleNet(Model):
    """
    https://arxiv.org/abs/1409.4842
    2014.9.17, Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
    params_size = 13,378,280
    """

    def __init__(self, output_channel=1000):
        class Conv2d_Relu(Layer):
            def __init__(self, out_channels, kernel_size, stride=1,
                         pad=0):
                super().__init__()
                self.conv = L.Conv2d(out_channels, kernel_size, stride, pad)

            def forward(self, x):
                return F.relu(self.conv(x))

        class Inception(Layer):
            def __init__(self, out1, proj3, out3, proj5, out5, proj_pool):
                super().__init__()
                self.conv1 = Conv2d_Relu(out1, kernel_size=1)
                self.proj3 = Conv2d_Relu(proj3, kernel_size=1)
                self.conv3 = Conv2d_Relu(out3, kernel_size=3, pad=1)
                self.proj5 = Conv2d_Relu(proj5, kernel_size=1)
                self.conv5 = Conv2d_Relu(out5, kernel_size=5, pad=2)
                self.projp = Conv2d_Relu(proj_pool, kernel_size=1)

            def forward(self, x):
                out1 = self.conv1(x)
                out3 = self.conv3(self.proj3(x))
                out5 = self.conv5(self.proj5(x))
                pool = self.projp(F.max_pooling(x, 3, stride=1, pad=1))
                y = F.concatenate((out1, out3, out5, pool), axis=1)
                return y

        super().__init__()
        self.conv1 = Conv2d_Relu(64, 7, stride=2, pad=3)

        self.conv2_reduce = Conv2d_Relu(64, 1)
        self.conv2 = Conv2d_Relu(192, 3, stride=1, pad=1)

        self.inc3a = Inception(64, 96, 128, 16, 32, 32)
        self.inc3b = Inception(128, 128, 192, 32, 96, 64)

        self.inc4a = Inception(192, 96, 208, 16, 48, 64)
        self.inc4b = Inception(160, 112, 224, 24, 64, 64)
        self.inc4c = Inception(128, 128, 256, 24, 64, 64)
        self.inc4d = Inception(112, 144, 288, 32, 64, 64)
        self.inc4e = Inception(256, 160, 320, 32, 128, 128)

        self.inc5a = Inception(256, 160, 320, 32, 128, 128)
        self.inc5b = Inception(384, 192, 384, 48, 128, 128)

        self.loss3_fc = L.Linear(output_channel)

        self.loss1_conv = Conv2d_Relu(128, 1)
        self.loss1_fc1 = L.Linear(1024)
        self.loss1_fc2 = L.Linear(output_channel)

        self.loss2_conv = Conv2d_Relu(128, 1)
        self.loss2_fc1 = L.Linear(1024)
        self.loss2_fc2 = L.Linear(output_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)
        x = F.local_response_normalization(x)
        x = self.conv2_reduce(x)
        x = self.conv2(x)
        x = F.local_response_normalization(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)

        x = self.inc3a(x)
        x = self.inc3b(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)

        x = self.inc4a(x)
        if Config.train:
            aux1 = F.average_pooling(x, kernel_size=5, stride=3)
            aux1 = self.loss1_conv(aux1)
            aux1 = F.reshape(aux1, (x.shape[0], -1))
            aux1 = F.relu(self.loss1_fc1(aux1))
            aux1 = F.dropout(aux1, 0.7)
            aux1 = self.loss1_fc2(aux1)
        x = self.inc4b(x)
        x = self.inc4c(x)
        x = self.inc4d(x)
        if Config.train:
            aux2 = F.average_pooling(x, kernel_size=5, stride=3)
            aux2 = self.loss2_conv(aux2)
            aux2 = F.reshape(aux2, (x.shape[0], -1))
            aux2 = F.relu(self.loss2_fc1(aux2))
            aux2 = F.dropout(aux2, 0.7)
            aux2 = self.loss2_fc2(aux2)

        x = self.inc4e(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)

        x = self.inc5a(x)
        x = self.inc5b(x)
        x = F.average_pooling(x, kernel_size=7, stride=1)

        x = F.dropout(x, 0.4)
        x = F.reshape(x, (x.shape[0], -1))
        x = self.loss3_fc(x)
        if Config.train:
            return aux1, aux2, x
        return x


class ResNet(Model):
    """
    https://arxiv.org/abs/1512.03385
    2015.10.10, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    params_size =
    """
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/resnet{}.npz'

    def __init__(self, n_layers=152, pretrained=False):
        super().__init__()

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        self.conv1 = L.Conv2d(64, 7, 2, 3)
        self.bn1 = L.BatchNorm()
        self.res2 = BuildingBlock(block[0], 64, 64, 256, 1)
        print(1111111111)
        self.res3 = BuildingBlock(block[1], 256, 128, 512, 2)
        self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2)
        self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2)
        self.fc6 = L.Linear(1000)

        if pretrained:
            weights_path = utils.get_file(ResNet.WEIGHTS_PATH.format(n_layers))
            self.load_weights_epoch(weights_path)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pooling(x, kernel_size=3, stride=2)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = _global_average_pooling_2d(x)
        x = self.fc6(x)
        return x


class ResNet152(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(152, pretrained)


class ResNet101(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(101, pretrained)


class ResNet50(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(50, pretrained)


def _global_average_pooling_2d(x):
    N, C, H, W = x.shape
    h = F.average_pooling(x, (H, W), stride=1)
    h = F.reshape(h, (N, C))
    return h


class BuildingBlock(Layer):
    def __init__(self, n_layers=None, in_channels=None, mid_channels=None,
                 out_channels=None, stride=None, downsample_fb=None):
        super().__init__()

        self.a = BottleneckA(in_channels, mid_channels, out_channels, stride,
                             downsample_fb)
        self._forward = ['a']
        for i in range(n_layers - 1):
            name = 'b{}'.format(i + 1)
            print(name)
            bottleneck = BottleneckB(out_channels, mid_channels)
            setattr(self, name, bottleneck)
            self._forward.append(name)

    def forward(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x


class BottleneckA(Layer):
    """A bottleneck layer that reduces the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        downsample_fb (bool): If this argument is specified as ``False``,
            it performs downsampling by placing stride 2
            on the 1x1 convolutional layers (the original MSRA ResNet).
            If this argument is specified as ``True``, it performs downsampling
            by placing stride 2 on the 3x3 convolutional layers
            (Facebook ResNet).
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, downsample_fb=False):
        super().__init__()
        # In the original MSRA ResNet, stride=2 is on 1x1 convolution.
        # In Facebook ResNet, stride=2 is on 3x3 convolution.
        stride_1x1, stride_3x3 = (1, stride) if downsample_fb else (stride, 1)

        self.conv1 = L.Conv2d(mid_channels, kernel_size=1, stride=stride_1x1, pad=0, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(mid_channels, kernel_size=3, stride=stride_3x3, pad=1, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L.Conv2d(out_channels, kernel_size=1, stride=1, pad=0, nobias=True)
        self.bn3 = L.BatchNorm()
        self.conv4 = L.Conv2d(out_channels, kernel_size=1, stride=stride, pad=0, nobias=True)
        self.bn4 = L.BatchNorm()

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))

        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleneckB(Layer):
    """A bottleneck layer that maintains the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
    """

    def __init__(self, in_channels, mid_channels):
        super().__init__()

        self.conv1 = L.Conv2d(mid_channels, 1, 1, 0, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(mid_channels, 3, 1, 1, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L.Conv2d(in_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


def main_LeNet():
    batch_size = 100
    epoch = 10
    transfrom = compose(
        [toOpencv(), opencv_resize((32, 32)), toArray(), toFloat(),
         z_score_normalize(mean=[125.30691805, 122.95039414, 113.86538318],
                           std=[62.99321928, 62.08870764, 66.70489964])])
    trainset = lijnn.datasets.MNIST(train=True, x_transform=transfrom)
    testset = lijnn.datasets.MNIST(train=False, x_transform=transfrom)

    train_loader = lijnn.iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = lijnn.iterators.iterator(testset, batch_size, shuffle=False)

    model = LeNet_5()
    optimizer = lijnn.optimizers.Adam().setup(model)

    if lijnn.cuda.gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()

    for i in range(epoch):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            y = model(x)
            loss = lijnn.functions.softmax_cross_entropy(y, t)
            acc = lijnn.functions.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += loss.data
            sum_acc += acc.data
            print(loss.data)
        print(f"epoch {i + 1}")
        print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
        sum_loss, sum_acc = 0, 0

        with no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = lijnn.functions.softmax_cross_entropy(y, t)
                acc = lijnn.functions.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')


def main_AlexNet():
    batch_size = 100
    epoch = 10
    transfrom = compose(
        [toOpencv(), opencv_resize(227), toArray(), toFloat(),
         z_score_normalize(mean=[125.30691805, 122.95039414, 113.86538318],
                           std=[62.99321928, 62.08870764, 66.70489964])])
    trainset = lijnn.datasets.CIFAR10(train=True, x_transform=transfrom)
    testset = lijnn.datasets.CIFAR10(train=False, x_transform=transfrom)
    train_loader = lijnn.iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = lijnn.iterators.iterator(testset, batch_size, shuffle=False)

    model = AlexNet(10)
    optimizer = lijnn.optimizers.Adam(alpha=0.0001).setup(model)

    if lijnn.cuda.gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()

    for i in range(epoch):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            y = model(x)
            loss = lijnn.functions.softmax_cross_entropy(y, t)
            acc = lijnn.functions.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += loss.data
            sum_acc += acc.data
            print(f"loss : {loss.data} accuracy {acc.data}")
        print(f"epoch {i + 1}")
        print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
        sum_loss, sum_acc = 0, 0

        with no_grad(), test_mode():
            for x, t in test_loader:
                y = model(x)
                loss = lijnn.functions.softmax_cross_entropy(y, t)
                acc = lijnn.functions.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')


def main_VGG16(model_load=False, name='default'):
    batch_size = 10
    epoch = 10
    transfrom = compose(
        [toOpencv(), opencv_resize(224), toArray(), toFloat(),
         z_score_normalize(mean=[125.30691805, 122.95039414, 113.86538318],
                           std=[62.99321928, 62.08870764, 66.70489964])])
    trainset = lijnn.datasets.CIFAR10(train=True, x_transform=transfrom)
    testset = lijnn.datasets.CIFAR10(train=False, x_transform=transfrom)
    train_loader = lijnn.iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = lijnn.iterators.iterator(testset, batch_size, shuffle=False)

    model = VGG16(output_channel=10)
    optimizer = lijnn.optimizers.Adam(alpha=0.0001).setup(model)
    start_epoch = model.load_weights_epoch(name=name) + 1 if model_load else 1
    if lijnn.cuda.gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()

    for i in range(start_epoch, epoch + 1):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            y = model(x)
            loss = lijnn.functions.softmax_cross_entropy(y, t)
            acc = lijnn.functions.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += loss.data
            sum_acc += acc.data
            print(f"loss : {loss.data} accuracy {acc.data}")
        print(f"epoch {i + 1}")
        print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
        sum_loss, sum_acc = 0, 0

        with no_grad(), test_mode():
            for x, t in test_loader:
                y = model(x)
                loss = lijnn.functions.softmax_cross_entropy(y, t)
                acc = lijnn.functions.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')

        model.save_weights_epoch(i, name)


def main_GoogleNet(model_load=False, name='default'):
    batch_size = 64
    epoch = 100
    transfrom = compose(
        [toOpencv(), opencv_resize(224), toArray(), toFloat(),
         z_score_normalize(mean=[129.30416561, 124.0699627, 112.43405006], std=[68.1702429, 65.39180804, 70.41837019])])
    trainset = lijnn.datasets.CIFAR100(train=True, x_transform=transfrom)
    testset = lijnn.datasets.CIFAR100(train=False, x_transform=transfrom)
    train_loader = lijnn.iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = lijnn.iterators.iterator(testset, batch_size, shuffle=False)

    model = GoogleNet(output_channel=100)
    optimizer = lijnn.optimizers.Adam(alpha=0.0001).setup(model)
    start_epoch = model.load_weights_epoch(name=name) + 1 if model_load else 1

    if lijnn.cuda.gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()

    for i in range(start_epoch, epoch + 1):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            aux1, aux2, y = model(x)

            loss1 = lijnn.functions.softmax_cross_entropy(aux1, t)
            loss2 = lijnn.functions.softmax_cross_entropy(aux2, t)
            loss3 = lijnn.functions.softmax_cross_entropy(y, t)
            loss = loss3 + 0.3 * (loss1 + loss2)
            acc = lijnn.functions.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += loss.data
            sum_acc += acc.data
            print(f"loss : {loss.data} accuracy {acc.data}")
        print(f"epoch {i}")
        print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
        sum_loss, sum_acc = 0, 0

        with no_grad(), test_mode():
            for x, t in test_loader:
                y = model(x)
                loss = lijnn.functions.softmax_cross_entropy(y, t)
                acc = lijnn.functions.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')

        model.save_weights_epoch(i, name)
