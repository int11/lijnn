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
    def __init__(self, train=True, year=2007, x_transform=None, t_transform=None, cut_index=None, around_context=True):
        super(VOC_SelectiveSearch, self).__init__(train, year, x_transform, t_transform)
        self.around_context = around_context
        count = datasets.load_cache_npz('VOC_SelectiveSearch', train=train)
        if count is not None:
            self.label = count
            return

        for i in range(VOCDetection.__len__(self)):
            img, labels, bboxs = VOCDetection.__getitem__(self, i)
            ssbboxs = utils.SelectiveSearch(img)
            sslabels = []
            for e, ssbbox in enumerate(ssbboxs):
                bb_iou = [utils.get_iou(ssbbox, bbox) for bbox in bboxs]
                indexM = np.argmax(bb_iou)
                sslabels.append(labels[indexM] if bb_iou[indexM] > 0.50 else 21)

            label = np.append(ssbboxs, np.array(sslabels).reshape(-1, 1), axis=1)
            label = np.pad(label, ((0, 0), (1, 0)), mode='constant', constant_values=i)
            label = label[:2000] if len(label) > 2000 else label
            self.label = np.append(self.label, label, axis=0)

        datasets.save_cache_npz({'label': self.label}, 'VOC_SelectiveSearch', train=train)
        if cut_index is not None:
            self.label = self.label[cut_index[0]:cut_index[1]]

    def __getitem__(self, index):
        temp = self.label[index]
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
        labels[21] = 'backgound'
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
            if (label != 21 and pos_lag < self.pos_neg_number[0]) or (label == 21 and neg_lag < self.pos_neg_number[1]):
                x.append(img)
                t.append(label)

                if label == 21:
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
    epoch = 10
    trainset = VOC_SelectiveSearch(x_transform=transforms.resize(224), around_context=False)
    train_loader = rcnniter(trainset)

    model = VGG16_RCNN()
    optimizer = optimizers.Adam(alpha=0.0001).setup(model)
    start_epoch = model.load_weights_epoch(name=name)
    if cuda.gpu_enable:
        model.to_gpu()
        train_loader.to_gpu()
        # test_loader.to_gpu()

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
