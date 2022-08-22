from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *


class BasicBlock(Model):
    expansion = 1

    def __init__(self, out_channel, stride, downsample):
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
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = self.bn2(self.conv2(h1))

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

    def __init__(self, out_channel, stride, downsample):
        super(Bottleneck, self).__init__()
        self.downsample = downsample

        self.conv1 = L.Conv2d(out_channel, kernel_size=1, stride=1, pad=0, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(out_channel, kernel_size=3, stride=stride, pad=1, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L.Conv2d(out_channel * self.expansion, kernel_size=1, stride=1, pad=0, nobias=True)
        self.bn3 = L.BatchNorm()
        if self.downsample:
            self.conv1_1 = L.Conv2d(out_channel * self.expansion, kernel_size=1, stride=stride, pad=0, nobias=True)
            self.bn1_1 = L.BatchNorm()

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))

        if self.downsample:
            x = self.conv1_1(x)
            x = self.bn1_1(x)

        x = F.relu(h1 + x)

        return x


class ResNet(Model):
    def __init__(self, block, num_layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.in_channel = 64

        self.conv1 = L.Conv2d(self.in_channel, kernel_size=7, stride=2, pad=3,
                              nobias=True)
        self.bn1 = L.BatchNorm()

        self.layer1 = self._make_layer(block, num_layers[0], 64, stride=1)
        self.layer2 = self._make_layer(block, num_layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, num_layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, num_layers[3], 512, stride=2)
        self.fc = L.Linear(num_classes)

    def _make_layer(self, block, num_blocks, out_channel, stride):
        downsample = False
        print(1, stride, self.in_channel, out_channel * block.expansion, end='   ')
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            print(2)
            downsample = True

        layers = []
        layers.append(block(out_channel, stride, downsample))
        for _ in range(num_blocks - 1):
            layers.append(block(out_channel, 1, False))

        self.in_channel = out_channel * block.expansion
        print()
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
    """
    ResNet-18 model from
    "Deep Residual Learning for Image Recognition"
    https://arxiv.org/abs/1512.03385
    2015.12.10, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=1000):
    """
    ResNet-34 model from
    "Deep Residual Learning for Image Recognition"
    https://arxiv.org/abs/1512.03385
    2015.12.10, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=1000):
    """
    ResNet-50 model from
    "Deep Residual Learning for Image Recognition"
    https://arxiv.org/abs/1512.03385
    2015.12.10, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    param_size = 25610152
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=1000):
    """
    ResNet-101 model from
    "Deep Residual Learning for Image Recognition"
    https://arxiv.org/abs/1512.03385
    2015.12.10, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=1000):
    """
    ResNet-152 model from
    "Deep Residual Learning for Image Recognition"
    https://arxiv.org/abs/1512.03385
    2015.12.10, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def main_ResNet(name='default'):
    batch_size = 32
    epoch = 100
    transfrom = compose(
        [toOpencv(), opencv_resize(224), toArray(), toFloat(),
         z_score_normalize(mean=[129.30416561, 124.0699627, 112.43405006], std=[68.1702429, 65.39180804, 70.41837019])])
    trainset = datasets.CIFAR100(train=True, x_transform=transfrom)
    testset = datasets.CIFAR100(train=False, x_transform=transfrom)
    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = iterators.iterator(testset, batch_size, shuffle=False)

    model = resnet50(num_classes=100)
    optimizer = optimizers.Adam(alpha=0.0001).setup(model)
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

        with no_grad(), test_mode():
            for x, t in test_loader:
                y = model(x)
                loss = functions.softmax_cross_entropy(y, t)
                acc = functions.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')

        model.save_weights_epoch(i, name)


resnet18()
resnet34()
resnet50()
resnet101()
resnet152()
