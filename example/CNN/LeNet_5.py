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


def main_LeNet_5(name='default'):
    batch_size = 100
    epoch = 10
    transfrom = compose(
        [toOpencv(), opencv_resize((32, 32)), toArray(), toFloat(),
         z_score_normalize(mean=[33.31842145], std=[78.56748998])])
    trainset = datasets.MNIST(train=True, x_transform=transfrom)
    testset = datasets.MNIST(train=False, x_transform=transfrom)

    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = iterators.iterator(testset, batch_size, shuffle=False)

    model = LeNet_5()
    optimizer = optimizers.Adam().setup(model)
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

        with no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = functions.softmax_cross_entropy(y, t)
                acc = functions.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')
        model.save_weights_epoch(i, name)