from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *

import cv2 as cv
import numpy as np


class VGG16(Model):
    """
    "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    https://arxiv.org/abs/1409.1556
    2014.9.4, Karen Simonyan, Andrew Zisserman
    params_size = 138,357,544

    """
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz'

    def __init__(self, pretrained=False, num_classes=1000):
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
        self.fc8 = L.Linear(num_classes)

        if pretrained:
            weights_path = utils.get_file(VGG16.WEIGHTS_PATH)
            self.load_weights(weights_path)

        self.conv6 = L.Conv2d(4096, kernel_size=7, stride=1, pad=0)
        self.conv7 = L.Conv2d(4096, kernel_size=1, stride=1, pad=0)
        self.conv8 = L.Conv2d(self.fc8.out_size, kernel_size=1, stride=1, pad=0)
        self._params.remove('conv6')
        self._params.remove('conv7')
        self._params.remove('conv8')
        # 512*7*7, 4096
        # 4096,512,7,7
        self.conv6.W.data = self.fc6.W.data.reshape((512, 7, 7, 4096)).transpose(3, 0, 1, 2)
        self.conv7.W.data = self.fc7.W.data.reshape((4096, 1, 1, 4096)).transpose(3, 0, 1, 2)
        self.conv8.W.data = self.fc8.W.data.reshape((4096, 1, 1, 1000)).transpose(3, 0, 1, 2)
        self.conv6.b.data, self.conv7.b.data, self.conv8.b.data = self.fc6.b.data, self.fc7.b.data, self.fc8.b.data

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
        # x.shape = (10, 512, 7, 7)
        if Config.train:
            x = F.reshape(x, (x.shape[0], -1))
            x = F.dropout(F.relu(self.fc6(x)))
            x = F.dropout(F.relu(self.fc7(x)))
            x = self.fc8(x)
        else:
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)
        return x

    def predict(self, x):
        pass

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        if size:
            image = cv.resize(image, size)
        image = image.astype(dtype)
        image -= np.array([103.939, 116.779, 123.68])
        image = image.transpose((2, 0, 1))
        return image


def main_VGG16(name='default'):
    batch_size = 10
    epoch = 10
    multi_scale_transform = compose(
        [random_isotropically_resize(256, 512), randomCrop(224), toFloat(),
         z_score_normalize(mean=[125.30691805, 122.95039414, 113.86538318],
                           std=[62.99321928, 62.08870764, 66.70489964])])
    trainset = datasets.CIFAR10(train=True, x_transform=multi_scale_transform)
    testset = datasets.CIFAR10(train=False, x_transform=multi_scale_transform)
    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    test_loader = iterators.iterator(testset, batch_size, shuffle=False)

    model = VGG16(num_classes=10)
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
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += loss.data
            sum_acc += acc.data
            print(f"loss : {loss.data} accuracy {acc.data}")
        print(f"epoch {i + 1}")
        print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
        model.save_weights_epoch(i, name)

        sum_loss, sum_acc = 0, 0
        with no_grad(), test_mode():
            for x, t in test_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')


"""
OverFeat dense evaluation방식을 vgg에 적용하므로써 연산량이 효율적이라고하셨는데 질문드립니다.

A모델은 마지막 layer이 3개가 fc-layer인 다들 아시는 vgg 기본 모델입니다.
B모델은 dense evaluation 적용을 위해 마지막 layer 3개가 conv layer 로 바꾼 vgg모델입니다.

example 1,
(225,225) 이미지를 상하좌우 crop 하여 4개의 이미지를 A모델에서 평가 합니다.
(225,225) 이미지를 그대로 넣어 B모델에서 dense 평가 합니다.
A모델은                               4개의 이미지를 평가합니다.
B모델은 size = 225-224+1, size*size = 4개의 이미지를 평가합니다.
input img size를 더 키워보겠습니다

example 2, 
(256,256) 이미지를 상하좌우,중간 crop 하여 5개의 이미지를 A모델에서 평가 합니다.
(256,256) 이미지를 그대로 넣어 B모델에서 dense 평가합니다.
A모델은                                 5 개의 이미지를 평가합니다.
B모델은 size = 256-224+1, size*size= 1089 개의 이미지를 평가합니다.

example 3, 
(256,256) 이미지를 stride = 1 pixel 기준으로 (224,224) size crop 하여 1089개의 이미지를 A모델에서 평가합니다.
(256,256) 이미지를 그대로 넣어 B모델에서 dense 평가합니다.
A,B 두 모델 모두 1089개의 이미지를 평가합니다.
여기서 질문이
글에서 말씀하신 "연산량이 효율적이다" 라는 의미는 모델안의 conv2d layer 이라던지 dense layer 의 연산횟수가 다름으로써 효율적이다가 아닌 
for문으로 crop과정이 생략되므로써 효율적이다 라는 뜻일까요?
근데 모델의 95% 정도 연산과정은 conv2d layer 이라던지 dense layer 의 연산에 쓰이는데 
dense evaluation쓰므로서 for문 crop과정을 생략하여 얻어지는 연산량의 효율은 그닥 없어보이는데 저의 이해한 방식이 맞을까요?
"""
