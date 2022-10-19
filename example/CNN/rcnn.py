import lijnn
import lijnn.functions as F
import numpy as np
import example.CNN as cnn
import cv2 as cv
import os
from sklearn.preprocessing import LabelBinarizer
import xml.etree.ElementTree as ET
from lijnn import datasets, utils

batch_size = 1
epoch = 10
trainset = datasets.VOCDetection()
train_loader = lijnn.iterators.iterator(trainset, batch_size, shuffle=True)
rect, labels, iou = utils.SelectiveSearch(*trainset[0])


def RCNN_batch(imgfile, rect, labels, pos_neg_number, idx, obj_class):
    train_images = []
    train_labels = []
    pos_lag = 0
    neg_lag = 0

    img_path = 'JPEGImages'

    Label_binarized = LabelBinarizer()
    Label_binarized.fit(obj_class)

    for i, filename in enumerate(reindexed_imgfile):
        if (reindexed_labels[i] != 'background' and pos_lag < pos_neg_number[0]) or (
                reindexed_labels[i] == 'background' and neg_lag < pos_neg_number[1]):

            image = cv.imread(os.path.join(img_path, filename + '.jpg'))
            x, y, w, h = reindexed_rect[i]

            cropped_arround_img = around_context(image, x, y, w, h, 16)
            cropped_arround_img = np.array(cropped_arround_img, dtype=np.uint8)
            resized = cv.resize(cropped_arround_img, (224, 224), interpolation=cv.INTER_AREA)

            train_images.append(resized)
            train_labels.append(reindexed_labels[i])

            if reindexed_labels[i] == 'background':
                neg_lag += 1
            elif reindexed_labels[i] != 'background':
                pos_lag += 1

        if pos_lag >= pos_neg_number[0] and neg_lag >= pos_neg_number[1]:
            pos_lag = 0
            neg_lag = 0
            sample_train_images = np.array(train_images)
            sample_train_labels = Label_binarized.transform(train_labels)
            train_images = []
            train_labels = []

            yield (sample_train_images, sample_train_labels)
