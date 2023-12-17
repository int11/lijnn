from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *


class AlexNet(Model):
    """
    "ImageNet Classification with Deep Convolutional Neural Networks"
    https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    2012, Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    params_size = 76,009,832

    use Relu activation function - Shoter Training Time
    use 2 Gpu

    use Max Pooling, Dropout, Local Response Normalization
    """

    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = L.Conv2d(96, kernel_size=11, stride=4, pad=0)
        self.conv2 = L.Conv2d(256, kernel_size=5, stride=1, pad=2)
        self.conv3 = L.Conv2d(384, kernel_size=3, stride=1, pad=1)
        self.conv4 = L.Conv2d(384, kernel_size=3, stride=1, pad=1)
        self.conv5 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(num_classes)

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

        x = F.flatten(x)
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    def predict(self, x, mean=0, std=1):
        xp = cuda.get_array_module(x)
        if x.ndim == 3:
            x = x[np.newaxis]

        transfrom = compose([isotropically_resize(259), centerCrop(259), toFloat(), z_score_normalize(mean, std)])
        x = xp.array([transfrom(i) for i in x])

        N, C, H, W = x.shape
        s = 227
        #corner and center crop
        result = [x[:, :, :s, :s], x[:, :, :s, W - s:], x[:, :, H - s:, :s], x[:, :, H - s:, W - s:],
                  xp.array([centerCrop(s)(i) for i in x])]
        result += [xp.flip(i, 3) for i in result]
        with no_grad(), test_mode():
            result = xp.array([F.softmax(self(i)).data for i in result])
        return xp.mean(result, axis=0)


def main_AlexNet(name='default'):
    batch_size = 128
    epoch = 10
    mean = [125.30691805, 122.95039414, 113.86538318]
    std = [62.99321928, 62.08870764, 66.70489964]
    trainset = datasets.CIFAR10(train=True, x_transform=compose(
        [isotropically_resize(259), centerCrop(259), randomCrop(227), randomFlip(), toFloat(),
         z_score_normalize(mean, std)]))

    testset = datasets.CIFAR10(train=False, x_transform=None)

    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = iterators.iterator(testset, batch_size, shuffle=False)

    model = AlexNet(10)
    optimizer = optimizers.Adam(alpha=0.0001).setup(model)
    model.fit(epoch, optimizer, train_loader, test_loader)


if __name__ == "__main__":
	main_AlexNet()