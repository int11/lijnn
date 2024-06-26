from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *
from functions import *
"""
"OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks"
https://arxiv.org/abs/1312.6229
2013.12.21, Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, Yann LeCun
"""


class OverFeat_accuracy(Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = L.Conv2d(96, kernel_size=7, stride=2, pad=0)
        self.conv2 = L.Conv2d(256, kernel_size=7, stride=1, pad=0)
        self.conv3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5 = L.Conv2d(1024, kernel_size=3, stride=1, pad=1)
        self.conv6 = L.Conv2d(1024, kernel_size=3, stride=1, pad=1)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(4096)
        self.fc9 = L.Linear(num_classes)

        self.conv7 = L.Conv2d_share_weight(4096, kernel_size=5, stride=1, pad=0, target=self.fc7)
        self.conv8 = L.Conv2d_share_weight(4096, kernel_size=1, stride=1, pad=0, target=self.fc8)
        self.conv9 = L.Conv2d_share_weight(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc9)
        self._params.remove('conv7')
        self._params.remove('conv8')
        self._params.remove('conv9')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pooling(x, kernel_size=3, stride=3)

        x = F.relu(self.conv2(x))
        x = F.max_pooling(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        x = F.relu(self.conv6(x))
        # subsampling ratio = 12
        if Config.train:
            x = F.max_pooling(x, kernel_size=3, stride=3)
            # receptive field = 77
            # subsampling ratio = 36
            x = F.reshape(x, (x.shape[0], -1))
            x = F.dropout(F.relu(self.fc7(x)))
            x = F.dropout(F.relu(self.fc8(x)))
            x = self.fc9(x)
        else:
            x = find_pooling(x, 3)
            # receptive field = 77
            # subsampling ratio = 36
            x = F.relu(self.conv7(x))
            x = F.relu(self.conv8(x))
            x = self.conv9(x)
        # receptive field = 221
        # subsampling ratio = 36
        return x

    def predict(self, x, mean=0, std=1):
        xp = cuda.get_array_module(x)
        if x.ndim == 3:
            x = x[np.newaxis]

        transfrom = compose([toFloat(), z_score_normalize(mean, std)])
        x = xp.array([transfrom(i) for i in x])

        result = [xp.array([resize(221 + 12 * 2)(i) for i in x]),
                  xp.array([resize((221 + 12 * 2 + 36 * 1, 221 + 12 * 2 + 36 * 2))(i) for i in x]),
                  xp.array([resize((221 + 12 * 2 + 36 * 4, 221 + 12 * 2 + 36 * 6))(i) for i in x]),
                  xp.array([resize((221 + 12 * 2 + 36 * 6, 221 + 12 * 2 + 36 * 9))(i) for i in x])]

        with no_grad(), test_mode():
            result = [F.softmax(self(i)).data for i in result]

        # list(scales), 3*3*N, num_classes, H, W
        result = xp.array([np.mean(i, (2, 3)) for i in result])
        # scales, 3*3*N, num_classes
        scales, N, num_classes = result.shape
        return xp.mean(result.reshape(scales, 3, 3, int(N / (3 * 3)), num_classes), (0, 1, 2))


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

        self.conv6 = L.Conv2d_share_weight(3072, kernel_size=6, stride=1, pad=0, target=self.fc6)
        self.conv7 = L.Conv2d_share_weight(4096, kernel_size=1, stride=1, pad=0, target=self.fc7)
        self.conv8 = L.Conv2d_share_weight(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc8)
        self._params.remove('conv6')
        self._params.remove('conv7')
        self._params.remove('conv8')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pooling(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pooling(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))
        x = F.max_pooling(x, kernel_size=2, stride=2)
        # receptive field = 71
        # subsampling ratio = 32

        if Config.train:
            x = F.flatten(x)
            x = F.dropout(F.relu(self.fc6(x)))
            x = F.dropout(F.relu(self.fc7(x)))
            x = self.fc8(x)
        else:
            x = F.relu(self.conv6(x))
            x = F.relu(self.conv7(x))
            x = self.conv8(x)
            # receptive field = 231
            # subsampling ratio = 32
        return x

# TODO Localization, object detection
    
def main_OverFeat(name='default'):
    batch_size = 64
    epoch = 100
    trainset = datasets.VOCclassfication(train=True)
    trainset.set_transforms("img", compose([isotropically_resize(256), centerCrop(256), randomCrop(221), randomFlip(), toFloat()]))

    testset = datasets.VOCclassfication(train=False)

    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = iterators.iterator(testset, 1, shuffle=False)

    model = OverFeat_accuracy(20)
    optimizer = optimizers.Adam(alpha=0.0001)
    model.fit(epoch, optimizer, train_loader, test_loader, name=name)

if __name__ == '__main__':
    main_OverFeat()


