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


def AroundContext(img, bbox, pad):
    image_mean = np.mean(img, axis=(1, 2))
    _, H, W = img.shape

    padded_image = np.full((H + 2 * pad, W + 2 * pad, 3), image_mean, dtype=np.uint8).transpose(2, 0, 1)
    padded_image[:, pad:(H + pad), pad:(W + pad)] = img

    return padded_image[:, bbox[1]:bbox[3] + 32, bbox[0]:bbox[2] + 32]


class VOC_SelectiveSearch(VOCclassfication):
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

            img = AroundContext(img, ssbbox, 16) if self.around_context else img[:, ssbbox[1]:ssbbox[3], ssbbox[0]:ssbbox[2]]

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



class rcnniter(lijnn.iterators):
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        super(rcnniter, self).__init__(dataset, batch_size, shuffle, gpu)


a = VOC_SelectiveSearch(around_context=False)
a.show(15662 + 6 - 1)

