import numpy as np


class Dense:
    def __init__(self,xlen, ylen, actifun, initialization=None):

        if initialization == 'Xavier':
            m = np.sqrt(6 / (xlen + ylen))
            self.w = np.random.uniform(-m, m, (xlen, ylen))
            self.b = np.random.uniform(-m, m, (1, ylen))
        elif initialization == 'He':
            m = np.sqrt(6 / xlen)
            self.w = np.random.uniform(-m, m, (xlen, ylen))
            self.b = np.random.uniform(-m, m, (1, ylen))
        elif initialization is None:
            self.w = np.random.uniform(-1, 1, (xlen, ylen))
            self.b = np.random.uniform(-1, 1, (1, ylen))
        self.actifun = actifun

    def __call__(self, x):
        return self.actifun(x, self.w, self.b)


class Dropout:
    def __init__(self, probability):
        self.probability = probability
