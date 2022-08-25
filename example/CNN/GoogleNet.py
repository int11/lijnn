from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *


class GoogleNet(Model):
    """
    "Going Deeper with Convolutions"
    https://arxiv.org/abs/1409.4842
    2014.9.17, Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
    params_size = 13,378,280
    """

    def __init__(self, num_classes=1000):
        class Conv2d_Relu(Layer):
            def __init__(self, out_channels, kernel_size, stride=1,
                         pad=0):
                super().__init__()
                self.conv = L.Conv2d(out_channels, kernel_size, stride, pad)

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

        self.loss3_fc = L.Linear(num_classes)

        self.loss1_conv = Conv2d_Relu(128, 1)
        self.loss1_fc1 = L.Linear(1024)
        self.loss1_fc2 = L.Linear(num_classes)

        self.loss2_conv = Conv2d_Relu(128, 1)
        self.loss2_fc1 = L.Linear(1024)
        self.loss2_fc2 = L.Linear(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)
        x = F.local_response_normalization(x)
        x = self.conv2_reduce(x)
        x = self.conv2(x)
        x = F.local_response_normalization(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)

        x = self.inc3a(x)
        x = self.inc3b(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)

        x = self.inc4a(x)
        if Config.train:
            aux1 = F.average_pooling(x, kernel_size=5, stride=3)
            aux1 = self.loss1_conv(aux1)
            aux1 = F.reshape(aux1, (x.shape[0], -1))
            aux1 = F.relu(self.loss1_fc1(aux1))
            aux1 = F.dropout(aux1, 0.7)
            aux1 = self.loss1_fc2(aux1)
        x = self.inc4b(x)
        x = self.inc4c(x)
        x = self.inc4d(x)
        if Config.train:
            aux2 = F.average_pooling(x, kernel_size=5, stride=3)
            aux2 = self.loss2_conv(aux2)
            aux2 = F.reshape(aux2, (x.shape[0], -1))
            aux2 = F.relu(self.loss2_fc1(aux2))
            aux2 = F.dropout(aux2, 0.7)
            aux2 = self.loss2_fc2(aux2)

        x = self.inc4e(x)
        x = F.max_pooling(x, kernel_size=3, stride=2, pad=1)

        x = self.inc5a(x)
        x = self.inc5b(x)
        x = F.average_pooling(x, kernel_size=7, stride=1)

        x = F.dropout(x, 0.4)
        x = F.reshape(x, (x.shape[0], -1))
        x = self.loss3_fc(x)
        if Config.train:
            return aux1, aux2, x
        return x


def main_GoogleNet(name='default'):
    batch_size = 64
    epoch = 100
    transfrom = compose(
        [toOpencv(), opencv_resize(224), toArray(), toFloat(),
         z_score_normalize(mean=[129.30416561, 124.0699627, 112.43405006], std=[68.1702429, 65.39180804, 70.41837019])])
    trainset = datasets.CIFAR100(train=True, x_transform=transfrom)
    testset = datasets.CIFAR100(train=False, x_transform=transfrom)
    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = iterators.iterator(testset, batch_size, shuffle=False)

    model = GoogleNet(num_classes=100)
    optimizer = optimizers.Adam(alpha=0.0001).setup(model)
    start_epoch = model.load_weights_epoch(name=name)

    if cuda.gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()

    for i in range(start_epoch, epoch + 1):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            aux1, aux2, y = model(x)

            loss1 = functions.softmax_cross_entropy(aux1, t)
            loss2 = functions.softmax_cross_entropy(aux2, t)
            loss3 = functions.softmax_cross_entropy(y, t)
            loss = loss3 + 0.3 * (loss1 + loss2)
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

        with no_grad(), test_mode():
            for x, t in test_loader:
                y = model(x)
                loss = functions.softmax_cross_entropy(y, t)
                acc = functions.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')

        model.save_weights_epoch(i, name)
