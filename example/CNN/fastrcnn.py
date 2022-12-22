import lijnn.datasets
from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *
from example.CNN import VGG16, rcnn


class Fast_R_CNN(VGG16):
    def __init__(self, num_classes=21):
        super().__init__(imagenet_pretrained=True)
        self.fc8 = L.Linear(num_classes)
        self.conv8 = L.Conv2d_share_weight(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc8)
        self._params.remove('conv8')
        self.Bbr = L.Linear(num_classes * 4)

    def forward(self, x, ssbboxs):
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
        # receptive field = 16
        # subsampling_ratio = 16
        x = F.roi_pooling(x, ssbboxs, 7, 1 / 16)
        # x.shape = (N, 512, 7, 7)
        x = F.flatten(x)
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))

        cls_score = self.fc8(x)
        bbox_pred = self.Bbr(x)
        return cls_score, bbox_pred.reshape(len(x), -1, 4)


class bbox_transpose:
    def __init__(self, img_outputsize):
        self.img_outputsize = img_outputsize

    def __call__(self, img_shape, bbox):
        H, W = img_shape[1:]
        OH, OW = pair(self.img_outputsize)
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * OW / W
        #bbox[[808, 1123, 1271]][:, [0,2]] * OW / W
        # if w == 0: x2 += 1
        bbox[bbox[:, 0] - bbox[:, 2] == 0, 2] += 1

        bbox[:, [1, 3]] = bbox[:, [1, 3]] * OH / H
        # if h == 0: y2 += 1
        bbox[bbox[:, 1] - bbox[:, 3] == 0, 3] += 1
        return bbox


class VOC_fastrcnn(rcnn.VOC_SelectiveSearch):
    def __init__(self, train=True, year=2007,
                 img_transform=compose([resize(224), toFloat(), z_score_normalize(Fast_R_CNN.mean, 1)]),
                 bbox_transform=bbox_transpose(224)):
        super(VOC_fastrcnn, self).__init__(train, year, img_transform)
        self.bbox_transform = bbox_transform

    def __getitem__(self, index):
        img = self._get_index_img(index)
        index = np.where(self.count[:, 0] == index)

        bbox, g = self.count[index][:, 1:], self.g[index]

        bbox[:, :4] = self.bbox_transform(img.shape, bbox[:, :4])
        g = self.bbox_transform(img.shape, g)
        return self.img_transform(img), bbox, self.iou[index], g

    def __len__(self):
        return super(lijnn.datasets.VOCclassfication, self).__len__()


def xy1xy2_to_xywh(xy1xy2):
    xp = cuda.get_array_module(xy1xy2)
    xmin, ymin, xmax, ymax = xp.split(xy1xy2, 4, axis=1)
    return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin


class Hierarchical_Sampling(lijnn.iterator):
    def __init__(self, dataset=VOC_fastrcnn(), N=2, R=128, positive_sample_per=0.25, shuffle=True, gpu=False):
        super(Hierarchical_Sampling, self).__init__(dataset, N, shuffle, gpu)
        self.r_n = round(R / N)
        self.positive_sample_per = positive_sample_per

    def __next__(self):
        xp = cuda.cupy if self.gpu else np

        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        img_batch, ssbboxs, label, t = [], [], [], []

        for i, (img, count, iou, g) in enumerate(batch):
            img_batch.append(img)
            positive_img_num = int(self.r_n * self.positive_sample_per)

            POSsample = np.where(iou >= 0.6)[0]
            #POSsample = POSsample[np.random.permutation(len(POSsample))[:positive_img_num]]

            NEGsample = np.where(~(iou >= 0.6))[0]
            #NEGsample = NEGsample[np.random.permutation(len(NEGsample))[:self.r_n - positive_img_num]]

            index = np.concatenate((POSsample, NEGsample))
            p, g = count[index][:, :4], g[index]
            ssbboxs.append(np.pad(p, ((0, 0), (1, 0)), mode='constant', constant_values=i))
            label.append(count[index][:, 4])

            p_x, p_y, p_w, p_h = xy1xy2_to_xywh(p)
            g_x, g_y, g_w, g_h = xy1xy2_to_xywh(g)
            t.append(np.concatenate([(g_x - p_x) / p_w, (g_y - p_y) / p_h, np.log(g_w / p_w), np.log(g_h / p_h)], axis=1))
        self.iteration += 1
        return (np.array(img_batch), np.concatenate(ssbboxs)), (np.concatenate(label), np.concatenate(t))


def multi_loss(x, x_bbox, t, t_bbox):
    loss_cls = F.softmax_cross_entropy(x, t)
    loss_loc = F.smooth_l1_loss(x_bbox, t_bbox)

    return loss_cls + 1 * loss_loc


def main_Fast_R_CNN(name='default'):
    batch_size = 16
    epoch = 10

    train_loader = Hierarchical_Sampling(shuffle=False)
    model = Fast_R_CNN()
    optimizer = optimizers.Adam(alpha=0.0001)
    model.fit(epoch, optimizer, train_loader, loss_function=multi_loss, name=name)
