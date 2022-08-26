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


class resize:
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, data):
        if data.dtype != np.uint8:
            raise ValueError("opencv.resize only supports uint8 type")

        data = cv.resize(data, (self.size[1], self.size[0]))
        if len(data.shape) == 2:
            data = data.reshape(data.shape + (1,))
        return data


class isotropically_resize:
    def __init__(self, S):
        self.S = S

    def __call__(self, data):
        argmin = np.argmin(data.shape[:-1])
        proportion = data.shape[argmin == 0] / data.shape[argmin]
        size = [self.S] * 2
        size[argmin == 0] = int(size[argmin == 0] * proportion)
        data = cv.resize(data, (size[1], size[0]))
        if len(data.shape) == 2:
            data = data.reshape(data.shape + (1,))
        return data


class toArray:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.transpose(2, 0, 1)


class toOpencv:
    def __call__(self, array):
        return array.transpose(1, 2, 0)


class randomHorizontalFlip:
    pass


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
