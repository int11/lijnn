from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *


class BasicBlock(Model):
    expansion = 1

    def __init__(self, out_channel, stride, downsample, base_width, groups):
        super().__init__()
        self.downsample = downsample

        self.conv1 = L.Conv2d(out_channel, kernel_size=3, stride=stride, pad=1, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(out_channel, kernel_size=3, stride=1, pad=1, nobias=True)
        self.bn2 = L.BatchNorm()
        if self.downsample:
            self.conv1_1 = L.Conv2d(out_channel * self.expansion, kernel_size=1, stride=stride, pad=0, nobias=True)
            self.bn1_1 = L.BatchNorm()

    def forward(self, x):
        h1 = self.conv1(x)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)

        h1 = self.conv2(h1)
        h1 = self.bn2(h1)

        if self.downsample:
            x = self.conv1_1(x)
            x = self.bn1_1(x)

        x = F.relu(h1 + x)

        return x


class Bottleneck(Model):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch

    expansion = 4

    def __init__(self, out_channel, stride, downsample, base_width, groups):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        width = int(out_channel * (base_width / 64.)) * groups

        self.conv1 = L.Conv2d(width, kernel_size=1, stride=1, pad=0, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(width, kernel_size=3, stride=stride, pad=1, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L.Conv2d(out_channel * self.expansion, kernel_size=1, stride=1, pad=0, nobias=True)
        self.bn3 = L.BatchNorm()
        if self.downsample:
            self.conv1_1 = L.Conv2d(out_channel * self.expansion, kernel_size=1, stride=stride, pad=0, nobias=True)
            self.bn1_1 = L.BatchNorm()

    def forward(self, x):
        h1 = self.conv1(x)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)

        h1 = self.conv2(h1)
        h1 = self.bn2(h1)
        h1 = F.relu(h1)

        h1 = self.conv3(h1)
        h1 = self.bn3(h1)

        if self.downsample:
            x = self.conv1_1(x)
            x = self.bn1_1(x)

        x = F.relu(h1 + x)

        return x


class ResNet(Model):
    def __init__(self, block, num_layers, num_classes=1000, base_width=64, groups=1):
        super(ResNet, self).__init__()

        self.out_channel = 64
        self.base_width = base_width
        self.groups = groups

        self.conv1 = L.Conv2d(self.out_channel, kernel_size=7, stride=2, pad=3,
                              nobias=True)
        self.bn1 = L.BatchNorm()

        self.layer1 = self._make_layer(block, num_layers[0], 64, stride=1)
        self.layer2 = self._make_layer(block, num_layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, num_layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, num_layers[3], 512, stride=2)
        self.fc = L.Linear(num_classes)

    def _make_layer(self, block, num_blocks, out_channel, stride):
        downsample = False
        if stride != 1 or self.out_channel != out_channel * block.expansion:
            downsample = True

        layers = []
        layers.append(block(out_channel, stride, downsample, self.base_width, self.groups))
        for _ in range(num_blocks - 1):
            layers.append(block(out_channel, 1, False, self.base_width, self.groups))

        self.out_channel = out_channel * block.expansion

        return models.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        def _global_average_pooling_2d(x):
            N, C, H, W = x.shape
            h = F.average_pooling(x, (H, W), stride=1)
            h = F.reshape(h, (N, C))
            return h

        x = _global_average_pooling_2d(x)
        x = self.fc(x)

        return x


def resnet18(num_classes=1000):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=1000):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=1000):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=1000):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=1000):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def resnext50_32x4d(num_classes=1000):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, 4, 32)


def resnext101_32x8d(num_classes=1000):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, 8, 32)


def wide_resnet50_2(num_classes=1000):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, 64 * 2)


def wide_resnet101_2(num_classes=1000):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, 64 * 2)
