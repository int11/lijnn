import lijnn.datasets
from lijnn import *
from lijnn.cuda import *
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
        x = F.roi_pooling(x, ssbboxs, 7, 1/16)
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
        bbox[:, 0] = np.floor(bbox[:, 0] * OW / W)
        bbox[:, 2] = np.ceil(bbox[:, 2] * OW / W)

        bbox[:, 1] = np.floor(bbox[:, 1] * OH / H)
        bbox[:, 3] = np.ceil(bbox[:, 3] * OH / H)

        return bbox


class VOC_fastrcnn(rcnn.VOC_SelectiveSearch):
    def __init__(self, train=True, year=2007,
                 img_transform=compose([resize(224), toFloat(), z_score_normalize(Fast_R_CNN.mean, 1)]),
                 bbox_transform=bbox_transpose(224)):
        super(VOC_fastrcnn, self).__init__(train, year, img_transform)
        self.bbox_transform = bbox_transform

    def __getitem__(self, index):
        img = self.getImg(index)
        index = np.where(self.count[:, 0] == index)

        bbox, g = self.count[index][:, 1:], self.g[index]

        if self.bbox_transform is not None:
            bbox[:, :4] = self.bbox_transform(img.shape, bbox[:, :4])
            g = self.bbox_transform(img.shape, g)
        return self.img_transform(img), bbox, self.iou[index].astype(np.float32), g

    def __len__(self):
        return super(lijnn.datasets.VOCclassfication, self).__len__()


def xy1xy2_to_xywh(xy1xy2):
    xp = cuda.get_array_module(xy1xy2)
    xmin, ymin, xmax, ymax = xp.split(xy1xy2, 4, axis=1)
    xmin, ymin, xmax, ymax = xmin.ravel(), ymin.ravel(), xmax.ravel(), ymax.ravel()
    return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin


def getT_from_P_G(p, g):
    xp = cuda.get_array_module(p)
    p_x, p_y, p_w, p_h = xy1xy2_to_xywh(p)
    g_x, g_y, g_w, g_h = xy1xy2_to_xywh(g)
    return Variable(xp.array([(g_x - p_x) / p_w, (g_y - p_y) / p_h, np.log(g_w / p_w), np.log(g_h / p_h)], dtype=np.float32).T)


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
        img_batch, ssbboxs = [], []
        label, p_batch, g_batch, u = [], [], [], []
        for i, (img, count, iou, g) in enumerate(batch):
            positive_img_num = int(self.r_n * self.positive_sample_per)

            POSindex = np.where(iou >= 0.6)[0]
            NEGindex = np.where(~(iou >= 0.6))[0]
            if self.shuffle:
                POSindex = POSindex[np.random.permutation(len(POSindex))]
                NEGindex = NEGindex[np.random.permutation(len(NEGindex))]
            POSindex = POSindex[:positive_img_num]
            NEGindex = NEGindex[:self.r_n - positive_img_num]

            index = np.concatenate((POSindex, NEGindex))

            p, g = count[index][:, :4], g[index]

            img_batch.append(img)
            ssbboxs.append(np.pad(p, ((0, 0), (1, 0)), mode='constant', constant_values=i))

            label.append(count[index][:, 4])
            p_batch.append(p)
            g_batch.append(g)
            u.append(np.ones_like(POSindex))
            u.append(np.zeros_like(NEGindex))

        self.iteration += 1
        return (xp.array(img_batch), xp.array(np.concatenate(ssbboxs))), \
               (xp.array(np.concatenate(label)), xp.array(np.concatenate(p_batch)), xp.array(np.concatenate(g_batch)),
                xp.array(np.concatenate(u)))


def multi_loss(y, y_bbox, t_label, p, g, u):
    xp = cuda.get_array_module(y)
    loss_cls = F.softmax_cross_entropy(y, t_label)

    y_bbox = y_bbox[xp.arange(len(y)), t_label]
    t_bbox = getT_from_P_G(p, g)
    u = u[None].T
    loss_loc = F.smooth_l1_loss(y_bbox * u, t_bbox * u)

    return loss_cls + loss_loc


def Faccuracy(y, y_bbox, t_label, p, g, u):
    xp = cuda.get_array_module(y)
    y, y_bbox, t_label, p, g, u = as_numpy(y), as_numpy(y_bbox), as_numpy(t_label), as_numpy(p), as_numpy(g), as_numpy(u)

    # acc
    acc = (y.argmax(axis=1) == t_label).mean()
    # iou
    y_bbox = y_bbox[np.arange(len(y)), t_label]
    index = u.astype(np.bool_)
    y_bbox, p = y_bbox[index], p[index]
    p_x, p_y, p_w, p_h = xy1xy2_to_xywh(p)
    x = p_w * y_bbox[:, 0] + p_x
    y = p_h * y_bbox[:, 1] + p_y
    w = p_w * np.exp(y_bbox[:, 2])
    h = p_h * np.exp(y_bbox[:, 3])
    y_bbox[:, 0], y_bbox[:, 1], y_bbox[:, 2], y_bbox[:, 3] = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    iou = sum([utils.IOU(a, b) for a, b in zip(y_bbox, g)]) / len(y_bbox)
    return {'acc': Variable(xp.array(acc)), 'iou': Variable(xp.array(iou))}


def main_Fast_R_CNN(name='default'):
    epoch = 10

    train_loader = Hierarchical_Sampling()
    model = Fast_R_CNN()
    optimizer = optimizers.Adam(alpha=0.0001)
    model.fit(epoch, optimizer, train_loader, loss_function=multi_loss, accuracy_function=Faccuracy,
              iteration_print=True, name=name)

if __name__ == "__main__":
	main_Fast_R_CNN()