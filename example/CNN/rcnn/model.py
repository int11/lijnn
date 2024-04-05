from example.CNN.vgg import VGG16
import lijnn.layers as L
import lijnn.functions as F


class Fast_R_CNN(VGG16):
    def __init__(self, num_classes=21):
        super().__init__(imagenet_pretrained=True)
        self.fc8 = L.Linear(num_classes)
        self.conv8 = L.Conv2d_share_weight(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc8)
        self.removeLayer('conv8')
        self.Bbr = L.Linear((num_classes + 1) * 4)

    def forward(self, x, bboxs):
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
        return x
        # x = roi_pooling(x, bboxs, 7, 1/16)
        # # x.shape = (N, 512, 7, 7)
        # x = F.flatten(x)
        # x = F.dropout(F.relu(self.fc6(x)))
        # x = F.dropout(F.relu(self.fc7(x)))

        # cls_score = self.fc8(x)
        # bbox_pred = self.Bbr(x)
        # return cls_score, bbox_pred.reshape(len(x), -1, 4)