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

    def __init__(self, num_classes=1000, imagenet_pretrained=False, dense_evaluate=False):
        super().__init__()
        self.dense_evaluate = dense_evaluate
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

        self.conv6 = L.share_weight_conv2d(4096, kernel_size=7, stride=1, pad=0, target=self.fc6)
        self.conv7 = L.share_weight_conv2d(4096, kernel_size=1, stride=1, pad=0, target=self.fc7)
        self.conv8 = L.share_weight_conv2d(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc8)
        self._params.remove('conv6')
        self._params.remove('conv7')
        self._params.remove('conv8')

        if imagenet_pretrained:
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
        # x.shape = (10, 512, 7, 7)
        if self.dense_evaluate:
            x = F.relu(self.conv6(x))
            x = F.relu(self.conv7(x))
            x = self.conv8(x)
        else:
            x = F.reshape(x, (x.shape[0], -1))
            x = F.dropout(F.relu(self.fc6(x)))
            x = F.dropout(F.relu(self.fc7(x)))
            x = self.fc8(x)
        return x

    # TODO: multi-crop & dense evaluation
    def predict(self, x, mean, std):
        xp = cuda.get_array_module(x)
        if x.ndim == 3:
            x = x[np.newaxis]

        transfrom = compose([toFloat(), z_score_normalize(mean, std)])
        x = xp.array([transfrom(i) for i in x])

        result = [xp.array([isotropically_resize(224 + 32 * 1)(i) for i in x]),
                  xp.array([isotropically_resize(224 + 32 * 5)(i) for i in x]),
                  xp.array([isotropically_resize(224 + 32 * 9)(i) for i in x])]
        with no_grad(), test_mode():
            temp = self.dense_evaluate
            self.dense_evaluate = True
            result = [F.softmax(self(i)).data for i in result]
            self.dense_evaluate = temp
        # list(scales), N, num_classes, H, W
        result = xp.array([np.mean(i, (2, 3)) for i in result])
        # scales, N, num_classes
        return xp.mean(result, axis=0)

    def predict_imagenet(self, x):
        # imagenet pretrain model train by BGR
        return self.predict(x[::-1], [103.939, 116.779, 123.68], 1)


def main_VGG16(name='default'):
    batch_size = 8
    epoch = 10
    mean = [125.30691805, 122.95039414, 113.86538318]
    std = [62.99321928, 62.08870764, 66.70489964]
    multi_scale_transform = compose(
        [random_isotropically_resize(256, 512), randomCrop(224), toFloat(), z_score_normalize(mean, std)])
    trainset = datasets.CIFAR10(train=True, x_transform=multi_scale_transform)
    testset = datasets.CIFAR10(train=False, x_transform=None)
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
                y = model.predict(x, mean, std)
                loss = F.categorical_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                sum_loss += loss.data
                sum_acc += acc.data
                print(f"loss : {loss.data} accuracy {acc.data}")
        print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')


"""
https://blog.naver.com/laonple/220749876381
https://blog.naver.com/laonple/220752877630
OverFeat dense evaluation방식을 vgg에 적용하므로써 연산량이 효율적이라고하셨는데 질문드립니다.

A모델은 마지막 layer이 3개가 fc-layer인 다들 아시는 vgg 기본 모델입니다.
B모델은 dense evaluation 적용을 위해 마지막 layer 3개가 conv layer 로 바꾼 vgg모델입니다.

example 1,
(225,225) 이미지를 상하좌우 multi-crop 하여 4개의 이미지를 A모델에서 평가 합니다.
(225,225) 이미지를 넣어 B모델에서 dense 평가 합니다.
A모델은                               4개의 이미지를 평가합니다.
B모델은 size = 225-224+1, size*size = 4개의 이미지를 평가합니다.
input img size를 더 키워보겠습니다

example 2, 
(256,256) 이미지를 상하좌우,중간 multi-crop 하여 5개의 이미지를 A모델에서 평가 합니다.
(256,256) 이미지를 넣어 B모델에서 dense 평가합니다.
A모델은                                 5 개의 이미지를 평가합니다.
B모델은 size = 256-224+1, size*size= 1089 개의 이미지를 평가합니다.

example 3, 
(256,256) 이미지를 stride = 1 pixel 기준으로 (224,224) size multi-crop 하여 1089개의 이미지를 A모델에서 평가합니다.
(256,256) 이미지를 넣어 B모델에서 dense 평가합니다.
A,B 두 모델 모두 1089개의 이미지를 평가합니다.
여기서 질문이
글에서 말씀하신 "연산량이 효율적이다" 라는 의미는 모델안의 conv2d layer 이라던지 dense layer 의 연산횟수가 다름으로써 효율적이다가 아닌 
for문으로 crop과정이 생략되므로써 효율적이다 라는 뜻일까요?
근데 모델의 95% 정도 연산과정은 conv2d layer 이라던지 dense layer 의 연산에 쓰이는데 
dense evaluation쓰므로서 for문 crop과정을 생략하여 얻어지는 연산량의 효율은 그닥 없어보이는데 저의 이해한 방식이 맞을까요?
"""

"""
실험 해보니깐 stride=2 maxpooling layer이 5개있어서. 224 -> 225 로 늘려도 마지막 픽셀하나가 잘리기 때문에  그게 총 5번. 행해지기 때문에
최소한 input img 2^5 = 32 픽셀을 늘려야 dense evaluation 최종 img size가 1 늘어나네요.

example 1, A모델 4개, B모델 1개 평가하는것이고 (input img 224 ~ 255 size 모두 앞서말한 pixel 잘림 이유로 모두 output으로 1개의 이미지만 나옴)
example 2, A 5개, B 4개 평가
example 3, A, 1089개, B 4개 평가

하는것으로 보이네요. 굳이 예시를 들면


example 4, 
(256,256) 이미지를 stride = 32 pixel 기준으로 (224,224) size multi-crop 하여 4개개의 이미지를 A모델에서 평가합니다.
(256,256) 이미지를 넣어 B모델에서 dense 평가합니다.
A,B 두 모델 모두 4개의 이미지를 평가합니다.


이게 그나마 비슷한 가정같은데 A,B 두 모델 layer forward 순차적 연산과정이 완벽히 똑같지도 않고 같은 의미를 가지지도 않아서 
"for 문 crop 과정" 만 달라진다는 저의 가정은 잘못됐네요
이런 연산과정의 차이가 엄밀이 무슨 의미를 가지는지 잘모르겠는데 나중에 실험해봐야겠네요..
"""


def _imagenet_pretrainmodel_test():
    url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
    img_path = lijnn.utils.get_file(url)
    img = cv.imread(img_path)
    model = VGG16(imagenet_pretrained=True)
    result = model.predict_imagenet(img.transpose(2, 0, 1)[::-1])
    predict_id = np.argmax(result)
    labels = lijnn.datasets.ImageNet.labels()
    print(labels[predict_id])


def _dense_evaluation_test():
    class VGG16_print(VGG16):
        def forward(self, x):
            print(data.shape)
            x = F.relu(self.conv1_1(x))
            x = F.relu(self.conv1_2(x))
            x = F.max_pooling(x, 2, 2)
            print(x.shape)
            x = F.relu(self.conv2_1(x))
            x = F.relu(self.conv2_2(x))
            x = F.max_pooling(x, 2, 2)
            print(x.shape)
            x = F.relu(self.conv3_1(x))
            x = F.relu(self.conv3_2(x))
            x = F.relu(self.conv3_3(x))
            x = F.max_pooling(x, 2, 2)
            print(x.shape)
            x = F.relu(self.conv4_1(x))
            x = F.relu(self.conv4_2(x))
            x = F.relu(self.conv4_3(x))
            x = F.max_pooling(x, 2, 2)
            print(x.shape)
            x = F.relu(self.conv5_1(x))
            x = F.relu(self.conv5_2(x))
            x = F.relu(self.conv5_3(x))
            x = F.max_pooling(x, 2, 2)
            print(x.shape)
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

    def preprocess(image, size=(224, 224), dtype=np.float32):
        if size:
            image = cv.resize(image, size)
        image = image.astype(dtype)
        image -= np.array([103.939, 116.779, 123.68])
        image = image.transpose((2, 0, 1))
        return image

    model = VGG16_print(imagenet_pretrained=True)

    labels = lijnn.datasets.ImageNet.labels()
    url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
    img_path = lijnn.utils.get_file(url)
    data = cv.imread(img_path)

    datali = [preprocess(data, size=(224, 224)), preprocess(data, size=(255, 255)),
              preprocess(data, size=(256, 256))]
    datali = [data[np.newaxis] for data in datali]

    print('\nB모델에 224, 255, 256 size img test \n')

    with no_grad(), test_mode():
        for data in datali:
            print(f'{data.shape[2:]} size img 평가 \n')
            y = model(data)
            print(f'{y.shape}       <-- {y.shape[2:]}개의 이미지 평가 \n')

            predict_id = np.argmax(y.data, axis=1).reshape(-1)
            print(predict_id, end='   ')
            for e in predict_id:
                print(labels[e], end=' ')
            print('\n\n\n\n')
