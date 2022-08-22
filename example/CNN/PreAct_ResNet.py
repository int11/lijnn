from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *


class PreActBasic(Model):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride):
        super().__init__()

        self.bn1 = L.BatchNorm()
        self.conv1 = L.Conv2d(out_channel, kernel_size=3, stride=stride, pad=1, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv2 = L.Conv2d(out_channel, kernel_size=3, stride=1, pad=1, nobias=True)

        if stride != 1 or in_channel != out_channel * self.expansion:
            self.conv1_1 = L.Conv2d(out_channel * self.expansion, kernel_size=1, stride=stride, pad=0, nobias=True)

    def forward(self, x):
        h1 = self.conv1(F.relu(self.bn1(x)))
        h1 = self.conv2(F.relu(self.bn2(h1)))

        if hasattr(self, 'conv1_1'):
            x = self.conv1_1(x)

        return h1 + x


class PreActBottleNeck(Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.bn1 = L.BatchNorm()
        self.conv1 = L.Conv2d(out_channels, kernel_size=1, stride=1, pad=0, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv2 = L.Conv2d(out_channels, kernel_size=3, stride=stride, pad=1, nobias=True)
        self.bn3 = L.BatchNorm()
        self.conv3 = L.Conv2d(out_channels * self.expansion, kernel_size=1, stride=1, pad=0, nobias=True)

        self.shortcut = models.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.conv1_1 = L.Conv2d(out_channels * self.expansion, kernel_size=1, stride=stride, pad=0, nobias=True)

    def forward(self, x):
        h1 = self.conv1(F.relu(self.bn1(x)))
        h1 = self.conv2(F.relu(self.bn2(h1)))
        h1 = self.conv3(F.relu(self.bn3(h1)))
        if hasattr(self, 'conv1_1'):
            x = self.shortcut(x)

        return h1 + x


class PreActResNet(Model):

    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()

        self.conv1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1, nobias=True)
        self.input_channels = 64
        self.bn1 = L.BatchNorm()

        self.layer1 = self._make_layers(block, num_block[0], 64, 1)
        self.layer2 = self._make_layers(block, num_block[1], 128, 2)
        self.layer3 = self._make_layers(block, num_block[2], 256, 2)
        self.layer4 = self._make_layers(block, num_block[3], 512, 2)

        self.fc = L.Linear(num_classes)

    def _make_layers(self, block, num_blocks, out_channel, stride):
        layers = [block(self.input_channels, out_channel, stride)]
        self.input_channels = out_channel * block.expansion

        for _ in range(num_blocks - 1):
            layers.append(block(self.input_channels, out_channel, 1))
            self.input_channels = out_channel * block.expansion

        return models.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

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


def preact_resnet18():
    return PreActResNet(PreActBasic, [2, 2, 2, 2])


def preact_resnet34():
    return PreActResNet(PreActBasic, [3, 4, 6, 3])


def preact_resnet50():
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3])


def preact_resnet101():
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3])


def preact_resnet152():
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3])
