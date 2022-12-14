import lijnn.optimizers
from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *
from lijnn.datasets import VOCDetection, VOCclassfication
from example.CNN import VGG16


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
        loaded = datasets.load_cache_npz(f'VOC_SelectiveSearch{year}', train=train)
        if loaded is not None:
            self.count, self.iou = loaded
        else:
            self.iou = [1.] * len(self.count)
            for i in range(VOCDetection.__len__(self)):
                img, labels, bboxs = VOCDetection.__getitem__(self, i)
                ssbboxs = utils.SelectiveSearch(img)[:2000]
                temp = []
                for ssbbox in ssbboxs:
                    bb_iou = [utils.IOU(ssbbox, bbox) for bbox in bboxs]
                    indexM = np.argmax(bb_iou)
                    temp.append(labels[indexM] if bb_iou[indexM] > 0.5 else 20)
                    self.iou.append(bb_iou[indexM])

                temp = np.append(ssbboxs, np.array(temp).reshape(-1, 1), axis=1)
                temp = np.pad(temp, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                self.count = np.append(self.count, temp, axis=0)
            self.iou = np.array(self.iou)

            # sort_index = np.apply_along_axis(lambda x: x[0], axis=1, arr=self.count).argsort()
            sort_index = np.empty(0, dtype=np.int32)
            for i in range(VOCDetection.__len__(self)):
                index = np.where(self.count[:, 0] == i)[0]
                sort_index = np.append(sort_index, index)

            self.count = self.count[sort_index]
            self.iou = self.iou[sort_index]
            datasets.save_cache_npz({'label': self.count, 'iou': self.iou}, f'VOC_SelectiveSearch{year}', train=train)

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
        super().__init__(imagenet_pretrained=True)
        self.pool5_feature = pool5_feature
        self.fc8 = L.Linear(num_classes)
        self.conv8 = L.Conv2d_share_weight(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc8)
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
            x = F.flatten(x)
            return x
        # x.shape = (10, 512, 7, 7)
        if self.dense_evaluate:
            x = F.relu(self.conv6(x))
            x = F.relu(self.conv7(x))
            x = self.conv8(x)
        else:
            x = F.flatten(x)
            x = F.dropout(F.relu(self.fc6(x)))
            x = F.dropout(F.relu(self.fc7(x)))
            x = self.fc8(x)
        return x


def main_VGG16_RCNN(name='default'):
    size = 3
    batch_size = size * 4
    epoch = 10
    model = VGG16_RCNN()
    trainset = VOC_SelectiveSearch(
        x_transform=compose([transforms.resize(224), toFloat(), z_score_normalize(model.mean, 1)]))
    train_loader = rcnniter(trainset, pos_neg_number=(size, size * 3))

    model.fit(epoch, lijnn.optimizers.Adam(alpha=0.00001), train_loader, name=name, iteration_print=True)


def trans_coordinate(c):
    xp = cuda.get_array_module(c)
    xmin, ymin, xmax, ymax = c
    x, y, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    return xp.array([x, y, w, h])


class VOC_Bbr(VOC_SelectiveSearch):
    def __init__(self, train=True, year=2007, img_transpose=None, around_context=True):
        super(VOC_Bbr, self).__init__(train, year, None, None, around_context)
        self.img_transpose = img_transpose

        index = np.where(self.iou > 0.6)
        self.p = self.count[index]
        self.g = []
        index = -1
        for e in self.p:
            if index != e[0]:
                index = e[0]
                label, bboxes = self._get_index_label_bboxes(e[0])
            bb_iou = [utils.IOU(e[1:5], bbox) for bbox in bboxes]
            indexM = np.argmax(bb_iou)
            self.g.append(bboxes[indexM])
        self.g = np.array(self.g)

        del self.count, self.iou

    def __getitem__(self, index):
        # xy1xy2
        p, g = self.p[index], self.g[index]
        index, p = p[0], p[1:5]
        img = self._get_index_img(index)
        img = AroundContext(img, p, 16) if self.around_context else img[:, p[1]:p[3], p[0]:p[2]]
        # xywh
        p, g = trans_coordinate(p), trans_coordinate(g)
        # xywh
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
        self.fc = L.Linear(4, nobias=True)

    def forward(self, x):
        if self.feature_model:
            x = self.feature_model(x)
        # xywh
        x = self.fc(x)
        return x

    def predict(self, pool5_feature, ssbboxs):
        xp = cuda.get_array_module(pool5_feature)
        # xywh
        d = self(pool5_feature).data
        # xywh
        p = xp.array([trans_coordinate(ssbbox) for ssbbox in ssbboxs])
        # xywh
        x, y, w, h = p[:, 2] * d[:, 0] + p[:, 0], p[:, 3] * d[:, 1] + p[:, 1], \
                     p[:, 2] * xp.exp(d[:, 2]), p[:, 3] * xp.exp(d[:, 3])
        # xy1xy2
        return xp.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.int32).T


def main_Bbr(name='default'):
    batch_size = 8
    epoch = 10
    vgg = VGG16_RCNN(pool5_feature=True)
    vgg.load_weights_epoch()
    if cuda.gpu_enable:
        vgg.to_gpu()

    trainset = VOC_Bbr(img_transpose=compose([resize(224), toFloat(), z_score_normalize(vgg.mean, 1)]))
    train_loader = iterators.iterator(trainset, batch_size, shuffle=True)
    model = Bounding_box_Regression(feature_model=vgg)
    model.fit(epoch, lijnn.optimizers.Adam(alpha=0.0001), train_loader, f_loss=F.mean_squared_error, f_accuracy=None,
              name=name, iteration_print=True)


class R_CNN(Model):
    def __init__(self):
        super(R_CNN, self).__init__()
        self.vgg = VGG16_RCNN()
        self.vgg.load_weights_epoch()
        self.Bbr = Bounding_box_Regression()
        self.Bbr.load_weights_epoch()

        self.trans_resize = compose([resize(224), toFloat(), z_score_normalize(self.vgg.mean, 1)])

    def forward(self, x):
        xp = cuda.get_array_module(x)
        ssbboxs = utils.SelectiveSearch(x)[:50]
        probs = xp.empty((0, 21))

        # batch memory issue
        for ssbbox in ssbboxs:
            img = AroundContext(x, ssbbox, 16)
            img = self.trans_resize(img)
            softmax = F.softmax(self.vgg(img[None]))
            probs = xp.append(probs, softmax.data, axis=0)

        except_background = xp.argmax(probs, axis=1) != 20
        ssbboxs, probs = ssbboxs[except_background], probs[except_background]

        index = utils.NMS(ssbboxs, probs, iou_threshold=0.1)
        if len(index) != 0:
            ssbboxs, probs = ssbboxs[index], probs[index]

        test = ssbboxs.copy()
        self.vgg.pool5_feature = True
        img = xp.array([self.trans_resize(AroundContext(x, ssbbox, 16)) for ssbbox in ssbboxs])
        pool5_feature = self.vgg(img)
        ssbboxs = self.Bbr.predict(pool5_feature, ssbboxs)
        self.vgg.pool5_feature = False

        ssbboxs[:, 0] = xp.maximum(0, ssbboxs[:, 0])
        ssbboxs[:, 1] = xp.maximum(0, ssbboxs[:, 1])
        ssbboxs[:, 2] = xp.minimum(x.shape[2], ssbboxs[:, 2])
        ssbboxs[:, 3] = xp.minimum(x.shape[1], ssbboxs[:, 3])
        return ssbboxs, xp.argmax(probs, axis=1), test


def main_R_CNN():
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

            bboxs, label, test = result
            img = cuda.as_numpy(img[::-1].transpose(1, 2, 0).copy())
            bboxs = cuda.as_numpy(bboxs)
            for i in bboxs:
                img = cv.rectangle(img, i[:2], i[2:], (255, 0, 0), 2)
            for i in test:
                img = cv.rectangle(img, i[:2], i[2:], (0, 255, 0), 2)
            cv.imshow('result', img)
            cv.waitKey()
