from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *
from example.CNN import VGG16


class Fast_R_CNN(VGG16):
    def __init__(self):
        super().__init__(imagenet_pretrained=True)

    def forward(self, x):
        ssbboxs = utils.SelectiveSearch(x)
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
        x = F.roi_pooling(x, 7, ssbboxs)
        # x.shape = (10, 512, 7, 7)
        if self.dense_evaluate:
            x = F.relu(self.conv6(x))
            x = F.relu(self.conv7(x))
            x = self.conv8(x)
        else:
            x = F.reshape(x, (x.shape[0], -1))
            x = F.dropout(F.relu(self.fc6(x)))
            x = F.dropout(F.relu(self.fc7(x)))
            x = self.fc8(x)
        return x
