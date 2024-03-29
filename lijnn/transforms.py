import numpy as np
from lijnn import cuda
from lijnn.utils import pair
import cv2 as cv
import random


class compose:
    """Compose several transforms.

    Args:
        transforms (list): list of transforms
    """

    def __init__(self, transforms=None):
        if transforms is None:
            transforms = []
        self.transforms = transforms

    def __call__(self, data):
        if not self.transforms:
            return data
        for t in self.transforms:
            data = t(data)
        return data


class cvtColor:
    def __init__(self, mode=cv.COLOR_BGR2RGB):
        self.mode = mode

    def __call__(self, data):
        return cv.cvtColor(data, self.mode)


def _lijnn_resize(data, size):
    if not (data.dtype == np.uint8 or data.dtype == np.float32):
        print(data.dtype)
        raise ValueError('opencv_resize dtype must be uint8 or float32')
    data = data.transpose(1, 2, 0)

    xp = cuda.get_array_module(data)
    data = cuda.as_numpy(data)

    data = cv.resize(data, (size[1], size[0]))
    if len(data.shape) == 2:
        data = data.reshape(data.shape + (1,))
    data = data.transpose(2, 0, 1)
    return xp.array(data)


class resize:
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, data):
        return _lijnn_resize(data, self.size)


class randomResize:
    def __init__(self, Hstart, Hend, Wstart=None, Wend=None):
        self.Hstart, self.Hend, self.Wstart, self.Wend = Hstart, Hend, Wstart, Wend

    def __call__(self, data):
        H = random.randrange(self.Hstart, self.Hend)
        W = random.randrange(self.Wstart, self.Wend) if self.Wstart is None else H
        size = (H, W)
        size = pair(size)
        return _lijnn_resize(data, size)


class isotropically_resize:
    '''
    H, W 중 큰 값을 선택해 S 값으로 변경하고 작은 값은 그에 따라 스케일링
    '''
    def __init__(self, S):
        self.S = S

    def __call__(self, data):
        H, W = data.shape[1], data.shape[2]
        # data.shape type is tuple
        argmin = np.argmin([H, W])

        if argmin == 1:
            proportion = H / W
            size = (int(self.S * proportion), self.S)
        else:
            proportion = W / H
            size = (self.S, int(self.S * proportion))
        # proportion = data.shape[argmin == 0] / data.shape[argmin]
        # size = [self.S] * 2
        # size[argmin == 0] = int(size[argmin == 0] * proportion)

        return _lijnn_resize(data, size)


class random_isotropically_resize:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __call__(self, data):
        S = random.randrange(self.start, self.end)
        return isotropically_resize(S)(data)


class toArray:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.transpose(2, 0, 1)


class toOpencv:
    def __call__(self, array):
        return array.transpose(1, 2, 0)


class z_score_normalize:
    """Normalize a NumPy array with mean and standard deviation.

    Args:
        mean (float or sequence): mean for all values or sequence of means for
         each channel.
        std (float or sequence):
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        mean, std = self.mean, self.std
        xp = cuda.get_array_module(data)
        if not xp.isscalar(mean):
            mshape = [1] * data.ndim
            mshape[0] = len(data) if len(self.mean) == 1 else len(self.mean)
            mean = xp.array(self.mean, dtype=data.dtype).reshape(*mshape)
        if not xp.isscalar(std):
            rshape = [1] * data.ndim
            rshape[0] = len(data) if len(self.std) == 1 else len(self.std)
            std = xp.array(self.std, dtype=data.dtype).reshape(*rshape)
        return (data - mean) / std


class flatten:
    def __call__(self, array):
        return array.flatten()


class astype:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


toFloat = astype


class toInt(astype):
    def __init__(self, dtype=np.int32):
        super().__init__(dtype)


class randomCrop:
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, data):
        C, H, W = data.shape
        KH, KW = self.size
        RH, RW = random.randrange(0, H - KH + 1), random.randrange(0, W - KW + 1)

        return data[:, RH:RH + KH, RW:RW + KW]


class centerCrop:
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, data):
        C, H, W = data.shape
        OH, OW = self.size
        left = (W - OW) // 2
        right = W - ((W - OW) // 2 + (W - OW) % 2)
        up = (H - OH) // 2
        bottom = H - ((H - OH) // 2 + (H - OH) % 2)
        return data[:, up:bottom, left:right]


class flip:
    def __init__(self, flipcode=2):
        self.flipcode = flipcode

    def __call__(self, data):
        xp = cuda.get_array_module(data)
        return xp.flip(data, self.flipcode)


class randomFlip:
    def __init__(self, flipcode=2, p=0.5):
        self.flipcode = flipcode
        self.p = p

    def __call__(self, data):
        xp = cuda.get_array_module(data)
        if random.randrange(0, 100) < self.p * 100:
            data = xp.flip(data, self.flipcode)
        return data
