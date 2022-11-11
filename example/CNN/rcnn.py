import cv2 as cv
import numpy as np

import lijnn.optimizers
from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *
from lijnn.datasets import VOCDetection, VOCclassfication
from example.CNN import VGG16
import xml.etree.ElementTree as ET
import time


def AroundContext(img, bbox, pad):
    xp = cuda.get_array_module(img)
    image_mean = xp.mean(img, axis=(1, 2))
    C, H, W = img.shape

    padded_image = xp.full((H + 2 * pad, W + 2 * pad, 3), image_mean, dtype=np.uint8).transpose(2, 0, 1)
    padded_image[:, pad:(H + pad), pad:(W + pad)] = img

    return padded_image[:, bbox[1]:bbox[3] + pad * 2, bbox[0]:bbox[2] + pad * 2]


class VOC_SelectiveSearch(VOCclassfication):
    def __init__(self, train=True, year=2007, x_transform=None, t_transform=None, around_context=True):
        super(VOC_SelectiveSearch, self).__init__(train, year, x_transform, t_transform)
        self.around_context = around_context
        loaded = datasets.load_cache_npz('VOC_SelectiveSearch', train=train)
        if loaded is not None:
            self.count = loaded[0]
        else:
            for i in range(VOCDetection.__len__(self)):
                img, labels, bboxs = VOCDetection.__getitem__(self, i)
                ssbboxs = utils.SelectiveSearch(img)
                temp = []
                for ssbbox in ssbboxs:
                    bb_iou = [utils.iou(ssbbox, bbox) for bbox in bboxs]
                    indexM = np.argmax(bb_iou)
                    temp.append(labels[indexM] if bb_iou[indexM] > 0.5 else 20)

                temp = np.append(ssbboxs, np.array(temp).reshape(-1, 1), axis=1)
                temp = np.pad(temp, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                temp = temp[:2000]
                self.count = np.append(self.count, temp, axis=0)
            self.count = self.count[np.apply_along_axis(lambda x: x[0], axis=1, arr=self.count).argsort()]
            datasets.save_cache_npz({'label': self.count}, 'VOC_SelectiveSearch', train=train)

    def __getitem__(self, index):
        temp = self.count[index]
        index, bbox, label = temp[0], temp[1:5], temp[5]
        img = self._get_index_img(index)
        img = AroundContext(img, bbox, 16) if self.around_context else img[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

        return self.x_transform(img), self.t_transform(label)

    @staticmethod
    def labels():
        labels = VOCclassfication.labels()
        labels[20] = 'background'
        return labels


class rcnniter(lijnn.iterator):
    def __init__(self, dataset, pos_neg_number=(32, 96), shuffle=True, gpu=False):
        batch_size = pos_neg_number[0] + pos_neg_number[1]
        super(rcnniter, self).__init__(dataset, batch_size, shuffle, gpu)
        self.pos_neg_number = pos_neg_number
        self.sindex = 0

    def __next__(self):
        xp = cuda.cupy if self.gpu else np

        x, t, pos_lag, neg_lag = [], [], 0, 0

        for i, index in enumerate(self.index[self.sindex:]):
            batch = self.dataset[index]
            img, label = batch[0], batch[1]
            if (label != 20 and pos_lag < self.pos_neg_number[0]) or (label == 20 and neg_lag < self.pos_neg_number[1]):
                x.append(img)
                t.append(label)
                if label == 20:
                    neg_lag += 1
                else:
                    pos_lag += 1
            if pos_lag >= self.pos_neg_number[0] and neg_lag >= self.pos_neg_number[1]:
                self.sindex += i + 1
                x = xp.array(x)
                t = xp.array(t)

                return x, t

        self.reset()
        raise StopIteration


class VGG16_RCNN(VGG16):
    def __init__(self, num_classes=21, pool5_feature=False):
        super().__init__(num_classes=1000, imagenet_pretrained=True, dense_evaluate=False)
        self.pool5_feature = pool5_feature
        self.fc8 = L.Linear(num_classes)
        self.conv8 = L.share_weight_conv2d(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc8)
        self._params.remove('conv8')

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
        if self.pool5_feature:
            x = F.reshape(x, (x.shape[0], -1))
            return x
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


def main_VGG16_RCNN(name='default'):
    size = 3
    batch_size = size * 4
    mean = [103.939, 116.779, 123.68]

    trainset = VOC_SelectiveSearch(
        x_transform=compose([transforms.resize(224), toFloat(), z_score_normalize(mean, 1)]))
    train_loader = rcnniter(trainset, pos_neg_number=(size, size * 3))
    model = VGG16_RCNN()
    model.fit(10, lijnn.optimizers.Adam(alpha=0.00001), train_loader, name=name, iteration_print=True)


def trans_coordinate(c):
    xmin, ymin, xmax, ymax = c
    x, y, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    return np.array([x, y, w, h])


class VOC_Bbr(VOC_SelectiveSearch):
    def __init__(self, train=True, year=2007, img_transpose=None, around_context=True):
        super(VOC_Bbr, self).__init__(train, year, None, None, around_context)
        self.img_transpose = img_transpose

        loaded = datasets.load_cache_npz('VOC_Bbr', train=train)
        if loaded is not None:
            self.p, self.g = loaded[0], loaded[1]
        else:
            self.count = self.count[np.where(self.count[:, 5] != 20)]

            index = []
            self.g = np.empty((0, 4), np.int32)
            for e in range(len(self.count)):
                label, bboxes = self._get_index_label_bboxes(self.count[e][0])
                bb_iou = [utils.iou(self.count[e][1:5], bbox) for bbox in bboxes]
                indexM = np.argmax(bb_iou)
                if bb_iou[indexM] > 0.6:
                    index.append(e)
                    self.g = np.vstack((self.g, bboxes[indexM]))
            self.p = self.count[:, :5][index]
            datasets.save_cache_npz({'x': self.p, 't': self.g}, 'VOC_Bbr', train=train)

        del self.count

    def __getitem__(self, index):
        p, g = self.p[index], self.g[index]
        index, p = p[0], p[1:]
        img = self._get_index_img(index)
        img = AroundContext(img, p, 16) if self.around_context else img[:, p[1]:p[3], p[0]:p[2]]

        p, g = trans_coordinate(p), trans_coordinate(g)
        t = np.array([(g[0] - p[0]) / p[2], (g[1] - p[1]) / p[3], np.log(g[2] / p[2]), np.log(g[3] / p[3])])
        return self.img_transpose(img), t

    def __len__(self):
        return len(self.p)


class Bounding_box_Regression(Model):
    def __init__(self, feature_model=None):
        super().__init__()
        self.feature_model = feature_model
        if self.feature_model:
            self._params.remove('feature_model')
        self.W_x = L.Linear(1, nobias=True)
        self.W_y = L.Linear(1, nobias=True)
        self.W_w = L.Linear(1, nobias=True)
        self.W_h = L.Linear(1, nobias=True)

    def forward(self, x):
        if self.feature_model:
            x = self.feature_model(x)
        d_x = self.W_x(x)
        d_y = self.W_y(x)
        d_w = self.W_w(x)
        d_h = self.W_h(x)
        xywh = F.concatenate((d_x, d_y, d_w, d_h), axis=1)
        return xywh

    def predict(self, pool5_feature, ssbbox):
        xp = cuda.get_array_module(pool5_feature)
        d_x, d_y, d_w, d_h = [i.data for i in self(pool5_feature)]
        p_x, p_y, p_w, p_h = trans_coordinate(ssbbox)
        x = p_w * d_x + p_x
        y = p_h * d_y + p_y
        w = p_w * xp.exp(d_w)
        h = p_h * xp.exp(d_h)

        return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)


def main_Bbr(name='default'):
    batch_size = 8
    mean = [103.939, 116.779, 123.68]
    vgg = VGG16_RCNN(pool5_feature=True)
    vgg.load_weights_epoch()

    trainset = VOC_Bbr(img_transpose=compose([resize(224), toFloat(), z_score_normalize(mean, 1)]))
    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    model = Bounding_box_Regression(feature_model=vgg)
    model.fit(10, lijnn.optimizers.Adam(alpha=0.0001), train_loader, f_loss=F.mean_squared_error, f_accuracy=None,
              name=name, iteration_print=True)


class R_CNN(Model):
    def __init__(self):
        super(R_CNN, self).__init__()
        self.vgg = VGG16_RCNN()
        self.vgg.load_weights_epoch()
        self.Bbr = Bounding_box_Regression()
        self.Bbr.load_weights_epoch()

    def forward(self, x):
        xp = cuda.get_array_module(x)
        ssbboxs = utils.SelectiveSearch(x)[:500]

        trans_resize = resize(224)
        probs = xp.empty((0, 21))
        for ssbbox in ssbboxs:
            img = AroundContext(x, ssbbox, 16)
            img = trans_resize(img)
            img = xp.expand_dims(img, axis=0)
            softmax = F.softmax(self.vgg(img))
            probs = xp.append(probs, softmax.data, axis=0)

        except_background = xp.argmax(probs, axis=1) != 20
        ssbboxs, probs = ssbboxs[except_background], probs[except_background]

        index = NMS(ssbboxs, probs, iou_threshold=0.1)
        if len(index) != 0:
            ssbboxs, probs = ssbboxs[index], probs[index]

        self.vgg.pool5_feature = True
        for i, ssbbox in enumerate(ssbboxs):
            img = AroundContext(x, ssbbox, 16)
            img = trans_resize(img)
            img = xp.expand_dims(img, axis=0)
            pool5_feature = self.vgg(img)
            ssbboxs[i] = xp.array(self.Bbr.predict(pool5_feature, ssbbox)).ravel()
        self.vgg.pool5_feature = False
        return ssbboxs, xp.argmax(probs, axis=1)


def NMS(bboxs, probs, iou_threshold=0.5):
    xp = cuda.get_array_module(probs)
    order = xp.max(probs, axis=1)
    order = order.argsort()[::-1]

    index = xp.array([True] * len(order))

    for i in range(len(order) - 1):
        ovps = utils.batch_iou(bboxs[order[i]], bboxs[order[i + 1:]])
        for j, ov in enumerate(ovps):
            if ov > iou_threshold:
                index[order[j + i + 1]] = False

    return index


if __name__ == '__main__':
    utils.printoptions()
    dataset = VOCDetection()
    loader = iterator(dataset, 1, shuffle=False)
    model = R_CNN()
    if cuda.gpu_enable:
        model.to_gpu()
        loader.to_gpu()

    with no_grad(), test_mode():
        for img, labels, bboxs in loader:
            img = img[0]
            result = model(img)

            bboxs, label = result
            img = cuda.as_numpy(img[::-1].transpose(1, 2, 0).copy())
            bboxs = cuda.as_numpy(bboxs)
            for i in bboxs:
                img = cv.rectangle(img, i[:2], i[2:], (255, 0, 0), 2)
            cv.imshow('result', img)
            cv.waitKey()
