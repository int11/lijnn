from INN import *
import INN
from INN import layers as L
from INN import functions as F
import cv2 as cv
import numpy as np
import os


class LeNet_5(Model):
    """
    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    https://ieeexplore.ieee.org/abstract/document/726791
    "Gradient-based learning applied to document recognition"
    1998, Yann LeCun Léon Bottou Yoshua Bengio Patrick Haffner

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


class AlexNet(Model):
    """
    https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    ImageNet Classification with Deep Convolutional Neural Networks
    2012, Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton

    use Relu activation function - Shoter Training Time
    use Max Pooling, Dropout, Local Response Normalization
    GPU use
    """

    def __init__(self, output_channel=1000):
        super().__init__()
        self.conv1 = L.Conv2d(96, kernel_size=11, stride=4, pad=0)
        self.batchnorm1 = L.BatchNorm()
        self.conv2 = L.Conv2d(256, kernel_size=5, stride=1, pad=2)
        self.batchnorm2 = L.BatchNorm()
        self.conv3 = L.Conv2d(384, kernel_size=3, stride=1, pad=1)
        self.conv4 = L.Conv2d(384, kernel_size=3, stride=1, pad=1)
        self.conv5 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(output_channel)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.local_response_normalization(x)
        # x = self.batchnorm1(x)
        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.relu(self.conv2(x))
        x = F.local_response_normalization(x)
        # x = self.batchnorm2(x)
        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x


class ZFNet(Model):
    """
    https://arxiv.org/abs/1311.2901
    Visualizing and Understanding Convolutional Networks
    2013, Matthew D Zeiler, Rob Fergus

    AlexNet -> ZFNet
    CONV1 : (11, 11) Kernel size, 4 strid -> (7, 7) Kernel size, 2 strid
    CONV3,4,5 : 384, 384, 256  out channels -> 512, 1024, 512 out channels
    """

    def __init__(self, output_channel=1000):
        super().__init__()
        self.conv1 = L.Conv2d(96, kernel_size=7, stride=2, pad=0)
        self.batchnorm1 = L.BatchNorm()
        self.conv2 = L.Conv2d(256, kernel_size=5, stride=1, pad=2)
        self.batchnorm2 = L.BatchNorm()
        self.conv3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4 = L.Conv2d(1024, kernel_size=3, stride=1, pad=1)
        self.conv5 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(output_channel)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.local_response_normalization(x)
        # x = self.batchnorm1(x)
        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.relu(self.conv2(x))
        x = F.local_response_normalization(x)
        # x = self.batchnorm2(x)
        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = F.max_pooling(x, kernel_size=3, stride=2)

        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x


class VGG16(Model):
    """
    https://arxiv.org/abs/1409.1556
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    2014, Karen Simonyan, Andrew Zisserman
    params_size = 138,357,544

    """
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz'

    def __init__(self, pretrained=False, output_channel=1000):
        super().__init__()
        self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(output_channel)

        if pretrained:
            weights_path = utils.get_file(VGG16.WEIGHTS_PATH)
            self.load_weights(weights_path)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        if size:
            image = cv.resize(image, size)
        image = image.astype(dtype)
        image -= np.array([103.939, 116.779, 123.68])
        image = image.transpose((2, 0, 1))
        return image


class GoogLeNet(Layer):
    """
    https://arxiv.org/abs/1409.4842
    2014, Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
    """
    def __init__(self):
        class Conv2d_Relu(Layer):
            def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0):
                super().__init__()
                self.conv = L.Conv2d(out_channels, kernel_size, stride=1,pad=0)
            def forward(self, x):
                return F.relu(self.conv(x))

        class Inception(Layer):
            def __init__(self, out1, proj3, out3, proj5, out5, proj_pool):
                super().__init__()
                self.conv1 = Conv2d_Relu(out1, kernel_size=1)
                self.proj3 = Conv2d_Relu(proj3, kernel_size=1)
                self.conv3 = Conv2d_Relu(out3, kernel_size=3, pad=1)
                self.proj5 = Conv2d_Relu(proj5, kernel_size=1)
                self.conv5 = Conv2d_Relu(out5, kernel_size=5, pad=2)
                self.projp = Conv2d_Relu(proj_pool, kernel_size=1)

            def forward(self, x):
                out1 = self.conv1(x)
                out3 = self.conv3(self.proj3(x))
                out5 = self.conv5(self.proj5(x))
                pool = self.projp(F.max_pooling(x, 3, stride=1, pad=1))
                y = F.concatenate((out1, out3, out5, pool), axis=1)
                return y

        super().__init__()
        self.conv1 = Conv2d_Relu(64, 7, stride=2, pad=3)
        self.conv2_reduce = Conv2d_Relu(64, 1)
        self.conv2 = Conv2d_Relu(192, 3, stride=1, pad=1)

        self.inc3a = Inception(64, 96, 128, 16, 32, 32)
        self.inc3b = Inception(128, 128, 192, 32, 96, 64)

        self.inc4a = Inception(192, 96, 208, 16, 48, 64)
        self.inc4b = Inception(160, 112, 224, 24, 64, 64)
        self.inc4c = Inception(128, 128, 256, 24, 64, 64)
        self.inc4d = Inception(112, 144, 288, 32, 64, 64)
        self.inc4e = Inception(256, 160, 320, 32, 128, 128)

        self.inc5a = Inception(256, 160, 320, 32, 128, 128)
        self.inc5b = Inception(384, 192, 384, 48, 128, 128)

        self.loss3_fc = L.Linear(1000)

        self.loss1_conv = L.Conv2d(512, 128, 1)
        self.loss1_fc1 = L.Linear(1024)
        self.loss1_fc2 = L.Linear(1000)

        self.loss2_conv = L.Conv2d(128, 1)
        self.loss2_fc1 = L.Linear(1024)
        self.loss2_fc2 = L.Linear(1000)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pooling(x,kernel_size=3,stride=2,pad=1)
        x = F.local_response_normalization(x)
        x = self.conv2_reduce(x)
        x = self.conv2(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)
        x = F.local_response_normalization(x)

        x = self.inc3a(x)
        x = self.inc3b(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)

        x = self.inc4a(x)
        x = self.inc4b(x)
        x = self.inc4c(x)
        x = self.inc4d(x)
        x = self.inc4e(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)

        x = self.inc5a(x)
        x = self.inc5b(x)
        x = F.average_pooling(x, kernel_size=7, stride=1)
        x = F.dropout(x, 0.4)
        x = F.reshape(x, (x.shape[0], -1))
        x = self.loss3_fc(x)

        return x


def main_LeNet():
    batch_size = 100
    epoch = 10
    transfrom = transforms.compose(
        [transforms.toOpencv(), transforms.opencv_resize((32, 32)), transforms.toArray(), transforms.toFloat(),
         transforms.z_score_Normalize(0., 255.)])
    trainset = INN.datasets.MNIST(train=True, x_transform=transfrom)
    testset = INN.datasets.MNIST(train=False, x_transform=transfrom)

    train_loader = INN.iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = INN.iterators.iterator(testset, batch_size, shuffle=False)

    model = LeNet_5()
    optimizer = INN.optimizers.Adam().setup(model)

    if INN.cuda.gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()

    for i in range(epoch):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            y = model(x)
            loss = INN.functions.softmax_cross_entropy(y, t)
            acc = INN.functions.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += loss.data
            sum_acc += acc.data
            print(loss.data)
        print(f"epoch {i + 1}")
        print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
        sum_loss, sum_acc = 0, 0

        with INN.no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = INN.functions.softmax_cross_entropy(y, t)
                acc = INN.functions.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')


def main_AlexNet():
    batch_size = 100
    epoch = 10
    transfrom = transforms.compose(
        [transforms.toOpencv(), transforms.opencv_resize(227), transforms.toArray(), transforms.toFloat(),
         transforms.z_score_Normalize(0.5, 0.5)])
    trainset = INN.datasets.CIFAR10(train=True, x_transform=transfrom)
    testset = INN.datasets.CIFAR10(train=False, x_transform=transfrom)
    train_loader = INN.iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = INN.iterators.iterator(testset, batch_size, shuffle=False)

    model = AlexNet()
    optimizer = INN.optimizers.Adam(alpha=0.0001).setup(model)

    if INN.cuda.gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()

    for i in range(epoch):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            y = model(x)
            loss = INN.functions.softmax_cross_entropy(y, t)
            acc = INN.functions.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += loss.data
            sum_acc += acc.data
            print(f"loss : {loss.data} accuracy {acc.data}")
        print(f"epoch {i + 1}")
        print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
        sum_loss, sum_acc = 0, 0

        with INN.test_mode():
            with INN.no_grad():
                for x, t in test_loader:
                    y = model(x)
                    loss = INN.functions.softmax_cross_entropy(y, t)
                    acc = INN.functions.accuracy(y, t)
                    sum_loss += loss.data
                    sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')


def main_VGG16():
    batch_size = 10
    epoch = 10
    transfrom = transforms.compose(
        [transforms.toOpencv(), transforms.opencv_resize(224), transforms.toArray(), transforms.toFloat(),
         transforms.z_score_Normalize(0.5, 0.5)])
    trainset = INN.datasets.CIFAR10(train=True, x_transform=transfrom)
    testset = INN.datasets.CIFAR10(train=False, x_transform=transfrom)
    train_loader = INN.iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = INN.iterators.iterator(testset, batch_size, shuffle=False)

    model = VGG16(output_channel=10)
    optimizer = INN.optimizers.Adam(alpha=0.0001).setup(model)

    if INN.cuda.gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()

    for i in range(epoch):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            y = model(x)
            loss = INN.functions.softmax_cross_entropy(y, t)
            acc = INN.functions.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += loss.data
            sum_acc += acc.data
            print(f"loss : {loss.data} accuracy {acc.data}")
        print(f"epoch {i + 1}")
        print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
        sum_loss, sum_acc = 0, 0

        with INN.test_mode():
            with INN.no_grad():
                for x, t in test_loader:
                    y = model(x)
                    loss = INN.functions.softmax_cross_entropy(y, t)
                    acc = INN.functions.accuracy(y, t)
                    sum_loss += loss.data
                    sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')

        weight_path = os.path.join(INN.utils.cache_dir, f'{model.__class__.__name__}{i}.npz')
        print(weight_path)
        model.save_weights(weight_path)
