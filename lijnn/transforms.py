import numpy as np
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
    if data.dtype != np.uint8:
        raise ValueError("opencv.resize only supports uint8 type")
    data = cv.resize(data, (size[1], size[0]))
    if len(data.shape) == 2:
        data = data.reshape(data.shape + (1,))
    return data


class resize:
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, data):
        return _lijnn_resize(data, self.size)


class randomResize:
    def __init__(self, Hstart, Hend, Wstart=None, Wend=None):
        self.a = 0 if Wstart is None else 1
        self.Hstart, self.Hend, self.Wstart, self.Wend = Hstart, Hend, Wstart, Wend

    def __call__(self, data):
        H = random.randrange(self.Hstart, self.Hend)
        W = random.randrange(self.Wstart, self.Wend) if self.a else H
        size = (H, W)
        size = pair(size)
        return _lijnn_resize(data, size)


class isotropically_resize:
    def __init__(self, S):
        self.S = S

    def __call__(self, data):
        argmin = np.argmin(data.shape[:-1])

        if argmin:
            proportion = data.shape[0] / data.shape[1]
            size = (int(self.S * proportion), self.S)
        else:
            proportion = data.shape[1] / data.shape[0]
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

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


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
    def __init__(self, dtype=np.int):
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
        H, W, C = data.shape
        OH, OW = self.size
        left = (W - OW) // 2
        right = W - ((W - OW) // 2 + (W - OW) % 2)
        up = (H - OH) // 2
        bottom = H - ((H - OH) // 2 + (H - OH) % 2)
        return data[:, up:bottom, left:right]


class randomFlip:
    """
    0 means flipping around the x-axis and positive value
    (for example, 1) means flipping around y-axis. Negative value
    (for example, -1) means flipping around both axes.
    """

    def __init__(self, flipcode=2, p=0.5):
        self.flipcode = flipcode
        self.p = p

    def __call__(self, data):
        if random.randrange(0, 100) < self.p * 100:
            data = np.flib(data, self.flipcode)
        return data
