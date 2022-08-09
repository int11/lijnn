import numpy as np
from lijnn import Layer
import lijnn.functions as F
import lijnn.layers as L
from lijnn import utils
import os


# =============================================================================
# Model / Sequential / MLP
# =============================================================================
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

    def save_weights(self, epoch, name='default'):
        model_dir = os.path.join(utils.cache_dir, self.__class__.__name__)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        weight_dir = os.path.join(model_dir, f'{name}_{epoch}_epoch.npz')

        super().save_weights(weight_dir)
        print(f'model weight save : {weight_dir}')

    def load_weights(self, epoch=None, name='default'):
        model_dir = os.path.join(utils.cache_dir, self.__class__.__name__)
        listdir = os.listdir(model_dir)

        name_listdir = [i for i in [i.split('_') for i in listdir] if i[0] == name]

        if epoch is None:
            epoch = max([i[1] for i in name_listdir])

        weight_dir = os.path.join(model_dir, f'{name}_{epoch}_epoch.npz')
        print(f'model weight load : {weight_dir}')
        super().load_weights(weight_dir)
        return int(epoch)


class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


# =============================================================================
# ResNet
# =============================================================================
class ResNet(Model):
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
        self.res3 = BuildingBlock(block[1], 256, 128, 512, 2)
        self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2)
        self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2)
        self.fc6 = L.Linear(1000)

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


class BuildingBlock(Layer):
    def __init__(self, n_layers=None, in_channels=None, mid_channels=None,
                 out_channels=None, stride=None, downsample_fb=None):
        super().__init__()

        self.a = BottleneckA(in_channels, mid_channels, out_channels, stride,
                             downsample_fb)
        self._forward = ['a']
        for i in range(n_layers - 1):
            name = 'b{}'.format(i + 1)
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

        self.conv1 = L.Conv2d(mid_channels, 1, stride_1x1, 0,
                              nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(mid_channels, 3, stride_3x3, 1,
                              nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L.Conv2d(out_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm()
        self.conv4 = L.Conv2d(out_channels, 1, stride, 0,
                              nobias=True)
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


# =============================================================================
# SqueezeNet
# =============================================================================
class SqueezeNet(Model):
    def __init__(self, pretrained=False):
        pass

    def forward(self, x):
        pass
