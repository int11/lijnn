import numpy as np
from lijnn import Variable
from lijnn import cuda
from lijnn.core import Function
from lijnn.utils import pair

class ROIPooling2D(Function):
    def __init__(self, output_size, spatial_scale):
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, x, bboxs):
        xp = cuda.get_array_module(x)
        assert bboxs.shape[-1] == 4, "bboxs input must be (N, bboxs_N, 4{ymin, xmin, ymax, xmax})"
        
        bboxs = bboxs.copy()

        _, C, H, W = x.shape
        OH, OW = pair(self.output_size)
        N, _ = bboxs.shape

        y = xp.zeros((N, C, OH, OW), dtype=x.dtype)
        self.argmax_data = xp.zeros(y.shape, xp.int32)

        bboxs[:, [0, 1]] = xp.floor(bboxs[:, [0, 1]] * self.spatial_scale)
        bboxs[:, [2, 3]] = xp.ceil(bboxs[:, [2, 3]] * self.spatial_scale)


        for n in range(N):
            ymin, xmin, ymax, xmax = bboxs[n, 0], bboxs[n, 1], bboxs[n, 2], bboxs[n, 3]
            bboxs_width, bboxs_height, stridew, strideh = ymax - ymin, xmax - xmin,  (ymax - ymin)/OW, (xmax - xmin)/OH

            for x in np.arange(0, bboxs_width, stridew):

                for y in np.arange(0, bboxs_height, strideh):
                    

        
        for n in range(N):
            for outh in range(OH):
                sliceh = L_sliceH[n][outh]
                for outw in range(OW):
                    slicew = L_sliceW[n][outw]
                    lenw = slicew[1] - slicew[0]

                    roi_data = x[int(bboxs[n][0]), :, sliceh[0]:sliceh[1], slicew[0]:slicew[1]].reshape(C, -1)

                    y[n, :, outh, outw] = xp.max(roi_data, axis=1)
                    index = xp.argmax(roi_data, axis=1)
                    ttt = (slicew[0] + sliceh[0] * W) + (index // lenw * W) + (index % lenw)
                    self.argmax_data[n, :, outh, outw] = ttt
        return y
    
    def backward(self, gy):
        x, bboxs = self.inputs
        gx, gbboxs = ROIPooling2DGrad(x.shape, self.argmax_data)(gy, bboxs)
        return gx, gbboxs


class ROIPooling2DGrad(Function):
    def __init__(self,  input_shape, argmax_data):
        self.input_shape = input_shape
        self.argmax_data = argmax_data
        

    def forward(self, gy, bboxs):
        xp = cuda.get_array_module(gy)

        _, C, H, W = self.input_shape
        N, _ = bboxs.shape

        gx = xp.zeros(self.input_shape, gy.dtype).ravel()
        a = bboxs[:,0] * C * H * W
        a = xp.broadcast_to(a.reshape(N,1,1,1), (N,C,1,1)) + (xp.arange(C) * H * W).reshape(1,C,1,1)
        a = self.argmax_data + a.reshape(N,C,1,1)
        a = a.ravel()
    
        gy_f = gy.ravel()
        if xp == np:
            np.add.at(gx, a, gy_f[np.arange(len(gy_f))])
        else:
            cuda.cpx.scatter_add(gx, a.ravel(), gy_f[np.arange(len(gy_f))])

        gx = gx.reshape(self.input_shape)

        return gx, None # gx, gbboxs

    def backward(self, ggx, ggbboxs):
        # No trivial way to implement double-backward for this function.
        raise NotImplementedError


def roi_pooling2d(x, rois, output_size, spatial_scale=1):
    return ROIPooling2D(output_size, spatial_scale)(x, rois)

if __name__ == "__main__":
    a = Variable(np.random.randn(1, 1, 5, 5).astype(np.float32))
    b = roi_pooling2d(a, np.array([[0, 0, 4, 4], [1, 1, 5, 5]]), 2, 1)
    b.backward()
    print(a.grad)