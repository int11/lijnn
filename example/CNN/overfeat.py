from lijnn import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *


class OverFeat(Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = L.Conv2d(96, kernel_size=7, stride=2, pad=0)
        self.conv2 = L.Conv2d(256, kernel_size=7, stride=1, pad=0)
        self.conv3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5 = L.Conv2d(1024, kernel_size=3, stride=1, pad=1)
        self.conv6 = L.Conv2d(1024, kernel_size=3, stride=1, pad=1)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(4096)
        self.fc9 = L.Linear(num_classes)

    def forward(self, x):
        print(x.shape)

        x = F.relu(self.conv1(x))
        x = F.max_pooling(x, kernel_size=3, stride=3)
        print(x.shape)

        x = F.relu(self.conv2(x))
        x = F.max_pooling(x, kernel_size=2, stride=2)
        print(x.shape)

        x = F.relu(self.conv3(x))
        print(x.shape)

        x = F.relu(self.conv4(x))
        print(x.shape)

        x = F.relu(self.conv5(x))
        print(x.shape)

        x = F.relu(self.conv6(x))
        x = F.max_pooling(x, kernel_size=3, stride=3)
        print(x.shape)

        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc7(x)))
        print(x.shape)
        x = F.dropout(F.relu(self.fc8(x)))
        print(x.shape)
        x = self.fc9(x)
        print(x.shape)
        return x


data = np.random.randint(1, 10, (1, 3, 245, 245))
model =OverFeat()
model(data)
print(model.params_size)