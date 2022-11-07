import cv2
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
    image_mean = np.mean(img, axis=(1, 2))
    _, H, W = img.shape

    padded_image = np.full((H + 2 * pad, W + 2 * pad, 3), image_mean, dtype=np.uint8).transpose(2, 0, 1)
    padded_image[:, pad:(H + pad), pad:(W + pad)] = img

    return padded_image[:, bbox[1]:bbox[3] + 32, bbox[0]:bbox[2] + 32]


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
                    bb_iou = [utils.get_iou(ssbbox, bbox) for bbox in bboxs]
                    indexM = np.argmax(bb_iou)
                    temp.append(labels[indexM] if bb_iou[indexM] > 0.5 else 20)

                temp = np.append(ssbboxs, np.array(temp).reshape(-1, 1), axis=1)
                temp = np.pad(temp, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                temp = temp[:2000] if len(temp) > 2000 else temp
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
    def __init__(self, num_classes=21):
        super().__init__(num_classes=1000, imagenet_pretrained=True)
        self.fc8 = L.Linear(num_classes)
        self.conv8 = L.share_weight_conv2d(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc8)
        self._params.remove('conv8')


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
    return x, y, w, h


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
            self.g = np.empty((1, 4), np.int32)
            for e in range(len(self.count)):
                label, bboxes = self._get_index_label_bboxes(self.count[e][0])
                bb_iou = [utils.get_iou(self.count[e][1:5], bbox) for bbox in bboxes]
                indexM = np.argmax(bb_iou)
                if bb_iou[indexM] > 0.6:
                    index.append(e)
                    self.g = np.vstack((self.g, bboxes[indexM]))
            self.g = np.delete(self.g, [0, 0], axis=0)
            self.p = self.count[:, :5][index]
            datasets.save_cache_npz({'x': self.p, 't': self.g}, 'VOC_Bbr', train=train)

        del self.count

    def __getitem__(self, index):
        p, g = self.p[index], self.g[index]
        index, p = p[0], p[1:]
        img = self._get_index_img(index)
        img = AroundContext(img, p, 16) if self.around_context else img[:, p[1]:p[3], p[0]:p[2]]
        return self.img_transpose(img), trans_coordinate(p), trans_coordinate(g)

    def __len__(self):
        return len(self.p)


class VGG16_pool5(VGG16_RCNN):
    def __init__(self, num_classes=21):
        super().__init__(num_classes)
        self.load_weights_epoch(classname="VGG16_RCNN")

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
        return x


class Bounding_box_Regression(Model):
    def __init__(self):
        super().__init__()
        self.W_x = L.Linear(1, nobias=True)
        self.W_y = L.Linear(1, nobias=True)
        self.W_w = L.Linear(1, nobias=True)
        self.W_h = L.Linear(1, nobias=True)

    def forward(self, x):
        d_x = self.W_x(x)
        d_y = self.W_y(x)
        d_w = self.W_w(x)
        d_h = self.W_h(x)

        return d_x, d_y, d_w, d_h

    def predict(self, x, ssbbox):
        d_x, d_y, d_w, d_h = self(x)
        p_x, p_y, p_w, p_h = trans_coordinate(ssbbox)
        pred_x = p_w * d_x + p_x
        pred_y = p_h * d_y + p_y
        pred_w = p_w * np.exp(d_w)
        pred_h = p_h * np.exp(d_h)

        return pred_x, pred_y, pred_w, pred_h

    def fit(self, feature_model, epoch, optimizer, train_loader, test_loader=None, lossf=F.mean_squared_error,
            name='default', iteration_print=False, autosave=True, autosave_time=30):
        optimizer = optimizer.setup(self)
        start_epoch, ti = self.load_weights_epoch(name=name)

        if cuda.gpu_enable:
            self.to_gpu()
            train_loader.to_gpu()
            if test_loader:
                test_loader.to_gpu()

        for i in range(start_epoch, epoch + 1):
            sum_loss, sum_acc = 0, 0
            st = time.time()
            for img, p, g in train_loader:
                with no_grad():
                    feature = feature_model(img)
                y = self(feature)
                loss = lossf(y, t)
                acc = F.accuracy(y, t)
                self.cleargrads()
                loss.backward()
                optimizer.update()
                sum_loss += loss.data
                sum_acc += acc.data
                if iteration_print:
                    print(f"loss : {loss.data} accuracy {acc.data}")
                if autosave and time.time() - st > autosave_time * 60:
                    self.save_weights_epoch(i, autosave_time + ti, name)
                    autosave_time += autosave_time
            print(f"epoch {i + 1}")
            print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
            self.save_weights_epoch(i, name=name)

            sum_loss, sum_acc = 0, 0
            if test_loader:
                with no_grad(), test_mode():
                    for x, t in test_loader:
                        y = self(x)
                        loss = F.softmax_cross_entropy(y, t)
                        acc = F.accuracy(y, t)
                        sum_loss += loss.data
                        sum_acc += acc.data
                        if iteration_print:
                            print(f"loss : {loss.data} accuracy {acc.data}")
                print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')


def main_Bbr(name='default'):
    batch_size = 8
    mean = [103.939, 116.779, 123.68]
    vggrcnn = VGG16_pool5()

    trainset = VOC_Bbr(img_transpose=compose([transforms.resize(224), toFloat(), z_score_normalize(mean, 1)]))
    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    model = Bounding_box_Regression()
    model.fit(vggrcnn, 10, lijnn.optimizers.Adam(alpha=0.001), train_loader, name=name, iteration_print=True)


if __name__ == '__main__':
    main_Bbr()
