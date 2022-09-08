from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *


class OverFeat_accuracy(Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = L.Conv2d(96, kernel_size=7, stride=2, pad=0)
        self.conv2 = L.Conv2d(256, kernel_size=7, stride=1, pad=0)
        self.conv3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5 = L.Conv2d(1024, kernel_size=3, stride=1, pad=1)
        self.conv6 = L.Conv2d(1024, kernel_size=3, stride=1, pad=1)
        self.fc7 = L.Conv2d(4096, kernel_size=5, stride=1, pad=0)
        self.fc8 = L.Conv2d(4096, kernel_size=1, stride=1, pad=0)
        self.fc9 = L.Conv2d(num_classes, kernel_size=1, stride=1, pad=0)

        self.conv7 = L.share_weight_conv2d(4096, kernel_size=5, stride=1, pad=0, target=self.fc7)
        self.conv8 = L.share_weight_conv2d(4096, kernel_size=1, stride=1, pad=0, target=self.fc8)
        self.conv9 = L.share_weight_conv2d(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc9)
        self._params.remove('conv7')
        self._params.remove('conv8')
        self._params.remove('conv9')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pooling(x, kernel_size=3, stride=3)
        print(x.shape)

        x = F.relu(self.conv2(x))
        x = F.max_pooling(x, kernel_size=2, stride=2)
        print(x.shape)

        x = F.relu(self.conv3(x))
        print(x.shape)

        x = F.relu(self.conv4(x))
        print(x.shape)

        x = F.relu(self.conv5(x))
        print(x.shape)

        x = F.relu(self.conv6(x))
        x = F.max_pooling(x, kernel_size=3, stride=3)
        # receptive field = 77
        print(x.shape)

        if Config.train:
            x = F.reshape(x, (x.shape[0], -1))
            x = F.dropout(F.relu(self.fc7(x)))
            print(x.shape)
            x = F.dropout(F.relu(self.fc8(x)))
            print(x.shape)
            x = self.fc9(x)
        else:
            x = F.relu(self.conv7(x))
            x = F.relu(self.conv8(x))
            x = self.conv9(x)
            # receptive field = 221
        print(x.shape)

        return x


class OverFeat_fast(Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = L.Conv2d(96, kernel_size=11, stride=4, pad=0)
        self.conv2 = L.Conv2d(256, kernel_size=5, stride=1, pad=0)
        self.conv3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4 = L.Conv2d(1024, kernel_size=3, stride=1, pad=1)
        self.conv5 = L.Conv2d(1024, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(3072)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(num_classes)

        self.conv6 = L.share_weight_conv2d(3072, kernel_size=6, stride=1, pad=0, target=self.fc6)
        self.conv7 = L.share_weight_conv2d(4096, kernel_size=1, stride=1, pad=0, target=self.fc7)
        self.conv8 = L.share_weight_conv2d(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc8)
        self._params.remove('conv6')
        self._params.remove('conv7')
        self._params.remove('conv8')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pooling(x, kernel_size=2, stride=2)
        print(x.shape)

        x = F.relu(self.conv2(x))
        x = F.max_pooling(x, kernel_size=2, stride=2)
        print(x.shape)

        x = F.relu(self.conv3(x))
        print(x.shape)

        x = F.relu(self.conv4(x))
        print(x.shape)

        x = F.relu(self.conv5(x))
        x = F.max_pooling(x, kernel_size=2, stride=2)
        # receptive field = 71
        print(x.shape)

        if Config.train:
            x = F.reshape(x, (x.shape[0], -1))
            x = F.dropout(F.relu(self.fc6(x)))
            print(x.shape)
            x = F.dropout(F.relu(self.fc7(x)))
            print(x.shape)
            x = self.fc8(x)
        else:
            x = F.relu(self.conv6(x))
            print(x.shape)
            x = F.relu(self.conv7(x))
            print(x.shape)
            x = self.conv8(x)
            # receptive field = 231
        print(x.shape)
        return x


