import lijnn
import lijnn.functions as F
import numpy as np
import example.CNN as cnn
import cv2 as cv
import os
from sklearn.preprocessing import LabelBinarizer
import xml.etree.ElementTree as ET
from lijnn import datasets, utils
from lijnn.datasets import VOCDetection, VOCclassfication
from lijnn import cuda
from glob import glob


def AroundContext(img, bbox, pad):
    image_mean = np.mean(img, axis=(1, 2))
    _, H, W = img.shape

    padded_image = np.full((H + 2 * pad, W + 2 * pad, 3), image_mean, dtype=np.uint8).transpose(2, 0, 1)
    padded_image[:, pad:(H + pad), pad:(W + pad)] = img

    return padded_image[:, bbox[1]:bbox[3] + 32, bbox[0]:bbox[2] + 32]


class VOC_SelectiveSearch1(VOCclassfication):
    def __init__(self, train=True, year=2007, x_transform=None, t_transform=None, cut_index=None, around_context=True):
        super(VOC_SelectiveSearch, self).__init__(train, year, None, None, cut_index)
        self.x_transform_trick = x_transform
        self.t_transform_trick = t_transform
        if self.x_transform_trick is None:
            self.x_transform_trick = lambda x: x
        if self.t_transform_trick is None:
            self.t_transform_trick = lambda x: x
        self.around_context = around_context
        self.ssbboxs = np.zeros((VOCDetection.__len__(self), 2000, 4), dtype=np.int32)

    def __getitem__(self, index):
        if len(self) <= index:
            raise IndexError
        elif VOCclassfication.__len__(self) > index:
            return VOCclassfication.__getitem__(self, index)
        else:
            index = index - VOCclassfication.__len__(self)
            a, b = divmod(index, 2000)
            img, labels, bboxs = VOCDetection.__getitem__(self, a)

            if np.sum(self.ssbboxs[a]) == 0:
                self.ssbboxs[a] = utils.SelectiveSearch(img)[:2000]
            ssbbox = self.ssbboxs[a][b]

            img = AroundContext(img, ssbbox, 16) if self.around_context else img[:, ssbbox[1]:ssbbox[3],
                                                                             ssbbox[0]:ssbbox[2]]

            bb_iou = [utils.get_iou(ssbbox, bbox) for bbox in bboxs]
            indexM = np.argmax(bb_iou)
            label = labels[indexM] if bb_iou[indexM] > 0.50 else 21

            return self.x_transform_trick(img), self.t_transform_trick(label)

    def __len__(self):
        return VOCclassfication.__len__(self) + VOCDetection.__len__(self) * 2000

    @staticmethod
    def labels():
        labels = VOCclassfication.labels()
        labels[21] = 'backgound'
        return labels


class VOC_SelectiveSearch(VOCclassfication):
    def __init__(self, train=True, year=2007, x_transform=None, t_transform=None, cut_index=None, around_context=True):
        super(VOC_SelectiveSearch, self).__init__(train, year, x_transform, t_transform, cut_index)
        for i in glob(f'{utils.cache_dir}/VOC_SelectiveSearch/*.txt'):
            with open(i, "r") as f:
                a = os.path.splitext(os.path.basename(i))[0]
                print(int(a))
                for line in f.readlines():
                    e = [int(i) for i in line.split()]
                    self.count = np.append(self.count, [[int(a), *e[:4], e[4]]])
        print()

    @staticmethod
    def labels():
        labels = VOCclassfication.labels()
        labels[21] = 'backgound'
        return labels


class rcnniter(lijnn.iterator):
    def __init__(self, dataset, pos_neg_number=(32, 96), shuffle=True, gpu=False):
        batch_size = pos_neg_number[0] + pos_neg_number[1]
        super(rcnniter, self).__init__(dataset, batch_size, shuffle, gpu)
        self.pos_neg_number = pos_neg_number
        self.sindex = 0

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        self.iteration += 1
        xp = cuda.cupy if self.gpu else np

        x = []
        t = []
        pos_lag = 0
        neg_lag = 0

        for i, index in enumerate(self.index[self.sindex:]):
            batch = self.dataset[index]
            img = batch[0]
            label = batch[1]
            if pos_lag < self.pos_neg_number[0] or neg_lag < self.pos_neg_number[1]:
                x.append(img)
                t.append(label)

                if label == 21:
                    neg_lag += 1
                elif label != 21:
                    pos_lag += 1
            else:
                self.sindex = i
                x = xp.array(x)
                t = xp.array(t)

                return x, t

dataset = VOC_SelectiveSearch()
