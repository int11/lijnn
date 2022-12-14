import lijnn.datasets
from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *
from example.CNN import VGG16, rcnn


class VOC_fastrcnn(rcnn.VOC_SelectiveSearch):
    def __init__(self, train=True, year=2007, x_transform=None, t_transform=None):
        super(VOC_fastrcnn, self).__init__(train, year, x_transform, t_transform)

    def __getitem__(self, index):
        img = self._get_index_img(index)
        temp = self.count[np.where(self.count[:, 0] == index)]
        ssbboxs, labels = temp[:, 1:5], temp[:, 5]
        return img, ssbboxs, labels,

    def __len__(self):
        return super(lijnn.datasets.VOCclassfication, self).__len__()


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
        return cls_score, bbox_pred


def multi_loss(x, x_bbox, t, t_bbox):
    loss_cls = F.softmax_cross_entropy(x, t)
    loss_loc = F.smooth_l1_loss(x_bbox, t_bbox)
    return loss_cls + loss_loc


def main_Fast_R_CNN():
    batch_size = 16
    epoch = 10
    model = Fast_R_CNN()
