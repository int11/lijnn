from lijnn import *
import numpy as np
import cupy as cp
from lijnn.functions import *

def test_add():
    x = Variable(np.array([3]))
    y = add(x, x)

    y.backward(retain_grad=True)

    print(y.grad.data == 1)
    print(x.grad.data == 2)


def test_roi_pooling():
    import torch
    import torch.nn as nn
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
    # x.to_gpu()
    y0 = roi_pooling(x0, cuda.get_array_module(x0.data).array([[0,0,0,4,4],[2,0,0,2,4]], dtype=np.int32), 3, 1)
    #print(y0)
    y0.backward()
    print(x0.grad.data)


    x = torch.from_numpy(data).type(torch.float)
    x.requires_grad=True
    y = SlowROIPool(3,1)(x, np.array([[0,0,0,4,4],[2,0,0,2,4]]))

    #print(y)
    y = y.sum()
    y.backward()
    print(x.grad.numpy())
    print((x0.grad.data != x.grad.numpy()).sum())

test_roi_pooling()