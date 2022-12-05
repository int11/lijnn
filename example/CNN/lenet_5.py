from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *


class LeNet_5(Model):
    """
    "Gradient-based learning applied to document recognition"
    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    https://ieeexplore.ieee.org/abstract/document/726791
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


def main_LeNet_5(name='default'):
    batch_size = 100
    epoch = 10
    transfrom = compose(
        [resize((32, 32)), toFloat(), z_score_normalize(mean=[33.31842145], std=[78.56748998])])
    trainset = datasets.MNIST(train=True, x_transform=transfrom)
    testset = datasets.MNIST(train=False, x_transform=transfrom)

    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = iterators.iterator(testset, batch_size, shuffle=False)

    model = LeNet_5()
    optimizer = optimizers.Adam().setup(model)
    model.fit(epoch, optimizer, train_loader, test_loader, name=name)
