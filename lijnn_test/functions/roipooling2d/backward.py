import numpy as np
from lijnn_test.functions.unit_test import f_unit_test, f_unit_test_withTorch
from example.CNN.rcnn.functions import roi_pooling2d



def test_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
    
    N, C, H, W = 1, 1, 5, 5
    OC, K = 1, 3

    rois = np.array([[0, 0, 4, 4], [1, 1, 5, 5]])

    input_data_shape = (N, C, H, W)

    kargs = {'output_size': 2}
    
    def lijnn_f(*args, **kwargs):
        return roi_pooling2d(rois=rois, *args, **kwargs)
    
    def torch_f(x, output_size, spatial_scale=1):
        return SlowROIPool(output_size, spatial_scale)(x, torch.tensor(rois))
    
    f_unit_test_withTorch(input_data_shape, torch_f, lijnn_f, **kargs)

if __name__ == "__main__":
    test_torch()
