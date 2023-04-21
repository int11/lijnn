import torch
import torch.nn as nn
from lijnn import *
import numpy as np
import cupy as cp
from lijnn.functions import *

gpu = False

class SlowROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.spatial_scale = spatial_scale
        self.size = output_size

    def forward(self, images, rois):
        rois[:, [1, 2]] = np.floor(rois[:, [1, 2]] * self.spatial_scale)
        rois[:, [3, 4]] = np.ceil(rois[:, [3, 4]] * self.spatial_scale)
        rois = rois.astype(int)

        n = rois.shape[0]
        x1 = rois[:, 1]
        y1 = rois[:, 2]
        x2 = rois[:, 3]
        y2 = rois[:, 4]

        res = []
        for i in range(n):
            img = images[int(rois[i][0])].unsqueeze(0)
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]]
            img = self.maxpool(img)
            res.append(img)
        res = torch.cat(res, dim=0)
        return res


np.random.seed(0)
data = np.random.randint(0,1000,(3,40,5,5)).astype(np.float32)

x0 = Variable(data)
if gpu:
    x0.to_gpu()
y0 = roi_pooling(x0, cuda.get_array_module(x0.data).array([[0,0,0,4,4],[2,0,0,2,4]], dtype=np.int32), 3, 1)
y0.backward()


x1 = torch.from_numpy(data).type(torch.float)
x1.requires_grad=True
y1 = SlowROIPool(3,1)(x1, np.array([[0,0,0,4,4],[2,0,0,2,4]]))
y1 = y1.sum()
y1.backward()
print(np.array_equal(x0.grad.data,x1.grad.numpy()))
