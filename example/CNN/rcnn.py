import lijnn.optimizers
from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *
from lijnn.datasets import VOCDetection, VOCclassfication
from example.CNN import VGG16


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
        count = datasets.load_cache_npz('VOC_SelectiveSearch', train=train)
        if count is not None:
            self.count = count
            return

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
        bytes = self.file.extractfile(self.image_tarinfo[index]).read()
        na = np.frombuffer(bytes, dtype=np.uint8)
        img = cv.imdecode(na, cv.IMREAD_COLOR)
        img = img.transpose(2, 0, 1)[::-1]
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


import xml.etree.ElementTree as ET


class VOC_Bbr(VOC_SelectiveSearch):
    def __init__(self, train=True, year=2007, x_transform=None, t_transform=None, around_context=True):
        super(VOC_Bbr, self).__init__(train, year, x_transform, t_transform, around_context)
        count = self.count[np.where(self.count[:, 5] != 20)]

        temp = []
        for e in range(len(count)):
            index = count[e][0]
            bytes = self.file.extractfile(self.xml_tarinfo[index]).read()
            annotation = ET.fromstring(bytes)
            bboxes = []
            for i in annotation.iter(tag="object"):
                budbox = i.find("bndbox")
                bboxes.append([int(budbox.find(i).text) for i in ['xmin', 'ymin', 'xmax', 'ymax']])

            a = [utils.get_iou(count[e][1:5], bbox) for bbox in bboxes]
            if np.max(a) > 0.6:
                temp.append(e)
        count = count[temp]
        print(count)

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
        p_x, p_y, p_w, p_h = (ssbbox[0] + ssbbox[2]) / 2, (ssbbox[1] + ssbbox[3]) / 2 \
            , ssbbox[2] - ssbbox[0], ssbbox[3] - ssbbox[1]
        pred_x = p_w * d_x + p_x
        pred_y = p_h * d_y + p_y
        pred_w = p_w * np.exp(d_w)
        pred_h = p_h * np.exp(d_h)

        return pred_x, pred_y, pred_w, pred_h


def main_Bbr():
    trainset = VOC_Bbr()


if __name__ == '__main__':
    main_Bbr()


def test():
    mean = [103.939, 116.779, 123.68]

    trainset = VOC_SelectiveSearch(x_transform=compose([transforms.resize(224)]), around_context=False)
    train_loader = rcnniter(trainset, pos_neg_number=(3, 3 * 3))
    for x, t in train_loader:
        for img, label in zip(x, t):
            cv.imshow('1', img[::-1].transpose(1, 2, 0))
            print(label)
            cv.waitKey(0)
