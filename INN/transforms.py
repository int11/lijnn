import numpy as np

try:
    import Image
except ImportError:
    from PIL import Image
from INN.utils import pair
import cv2 as cv


class Compose:
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


# =============================================================================
# Transforms for PIL Image
# =============================================================================
class cvtColor:
    def __init__(self, mode=cv.COLOR_BGR2RGB):
        self.mode = mode

    def __call__(self, img):
        return cv.cvtColor(img, self.mode)


class Resize:
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, img):
        if img.shape[2] == 1:
            img = cv.resize(img, self.size)
            return img.reshape(self.size + (1,))
        return cv.resize(img, self.size)


class CenterCrop:
    """Resize the input PIL image to the given size.

    Args:
        size (int or (int, int)): Desired output size.
        mode (int): Desired interpolation.
    """

    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, img):
        W, H = img.size
        OW, OH = self.size
        left = (W - OW) // 2
        right = W - ((W - OW) // 2 + (W - OW) % 2)
        up = (H - OH) // 2
        bottom = H - ((H - OH) // 2 + (H - OH) % 2)
        return img.crop((left, up, right, bottom))


class ToArray:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.transpose(2, 0, 1)


class ToOpencv:
    def __call__(self, array):
        return array.transpose(1, 2, 0)


class RandomHorizontalFlip:
    pass


# =============================================================================
# Transforms for NumPy ndarray
# =============================================================================
class Normalize:
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


class Flatten:
    def __call__(self, array):
        return array.flatten()


class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype=np.int):
        super().__init__(dtype)
