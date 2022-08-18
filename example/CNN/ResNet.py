from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *

import cv2 as cv
import numpy as np
import os


class BottleneckA(Layer):
    def __init__(self, mid_channels, out_channels, stride=2):
        super().__init__()

        self.conv1 = L.Conv2d(mid_channels, kernel_size=1, stride=stride, pad=0, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(mid_channels, kernel_size=3, stride=1, pad=1, nobias=True)
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
    def __init__(self, mid_channels, out_channels):
        super().__init__()

        self.conv1 = L.Conv2d(mid_channels, 1, 1, 0, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(mid_channels, 3, 1, 1, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L.Conv2d(out_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


class BuildingBlock(Layer):
    def __init__(self, n_layers, mid_channels, out_channels, stride):
        super().__init__()

        self.a = BottleneckA(mid_channels, out_channels, stride)
        self._forward = ['a']
        for i in range(n_layers - 1):
            name = f'b{i + 1}'
            setattr(self, name, BottleneckB(mid_channels, out_channels))
            self._forward.append(name)

    def forward(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)

        return x


class ResNet(Model):
    """
    https://arxiv.org/abs/1512.03385
    2015.10.10, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    params_size = {n_layer_50 : 23766116, n_layer_101 : 42810468, n_layer_152 : 58500196}
    """
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/resnet{}.npz'

    def __init__(self, n_layers=50, output_channel=1000, pretrained=False):
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
        self.res2 = BuildingBlock(block[0], 64, 256, 1)
        self.res3 = BuildingBlock(block[1], 128, 512, 2)
        self.res4 = BuildingBlock(block[2], 256, 1024, 2)
        self.res5 = BuildingBlock(block[3], 512, 2048, 2)
        self.fc6 = L.Linear(output_channel)

        if pretrained:
            weights_path = utils.get_file(ResNet.WEIGHTS_PATH.format(n_layers))
            self.load_weights(weights_path)

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


def main_ResNet(n_layers=50, name='default'):
    batch_size = 32
    epoch = 100
    transfrom = compose(
        [toOpencv(), opencv_resize(224), toArray(), toFloat(),
         z_score_normalize(mean=[129.30416561, 124.0699627, 112.43405006], std=[68.1702429, 65.39180804, 70.41837019])])
    trainset = datasets.CIFAR100(train=True, x_transform=transfrom)
    testset = datasets.CIFAR100(train=False, x_transform=transfrom)
    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = iterators.iterator(testset, batch_size, shuffle=False)

    model = ResNet(n_layers, output_channel=100)
    optimizer = optimizers.Adam(alpha=0.0001).setup(model)
    start_epoch = model.load_weights_epoch(name=name)

    if cuda.gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()

    for i in range(start_epoch, epoch + 1):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            y = model(x)
            loss = functions.softmax_cross_entropy(y, t)
            acc = functions.accuracy(y, t)
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
                loss = functions.softmax_cross_entropy(y, t)
                acc = functions.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')

        model.save_weights_epoch(i, name)
