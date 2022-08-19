from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *


class PreActBasic(Model):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.bn1 = L.BatchNorm()
        self.conv1 = L.Conv2d(out_channels, kernel_size=3, stride=stride, pad=1)
        self.bn2 = L.BatchNorm()
        self.conv2 = L.Conv2d(out_channels * PreActBasic.expansion, kernel_size=3, pad=1)

        self.shortcut = models.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = L.Conv2d(out_channels * PreActBasic.expansion, 1, stride=stride)

    def forward(self, x):
        h1 = self.conv1(F.relu(self.bn1(x)))
        h1 = self.conv2(F.relu(self.bn2(h1)))

        x = self.shortcut(x)

        return h1 + x


class PreActBottleNeck(Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.bn1 = L.BatchNorm()
        self.conv1 = L.Conv2d(out_channels, kernel_size=1, stride=stride, pad=0)
        self.bn2 = L.BatchNorm()
        self.conv2 = L.Conv2d(out_channels * PreActBottleNeck.expansion, kernel_size=3, stride=1, pad=1)
        self.bn3 = L.BatchNorm()
        self.conv3 = L.Conv2d(out_channels, kernel_size=1, stride=1, pad=0)

        self.shortcut = models.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = L.Conv2d(out_channels * PreActBottleNeck.expansion, 1, stride=stride)

    def forward(self, x):
        h1 = self.conv1(F.relu(self.bn1(x)))
        h1 = self.conv2(F.relu(self.bn2(h1)))
        h1 = self.conv3(F.relu(self.bn3(h1)))

        x = self.shortcut(x)

        return h1 + x


class PreActResNet(Model):

    def __init__(self, block, num_block, class_num=100):
        super().__init__()
        self.input_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_layers(block, num_block[0], 64, 1)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2)

        self.linear = nn.Linear(self.input_channels, class_num)

    def _make_layers(self, block, block_num, out_channels, stride):
        layers = []

        layers.append(block(self.input_channels, out_channels, stride))
        self.input_channels = out_channels * block.expansion

        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1))
            self.input_channels = out_channels * block.expansion
            block_num -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


def preactresnet18():
    return PreActResNet(PreActBasic, [2, 2, 2, 2])


def preactresnet34():
    return PreActResNet(PreActBasic, [3, 4, 6, 3])


def preactresnet50():
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3])


def preactresnet101():
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3])


def preactresnet152():
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3])
