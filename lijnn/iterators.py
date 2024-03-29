import math

pil_available = True
import numpy as np
from lijnn import cuda


class iterator:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        xp = cuda.cupy if self.gpu else np

        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        result = {}
        for key in batch[0].keys():
            for i in batch:
                result[key] = xp.stack([d[key] for d in batch])

        self.iteration += 1
        return result

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True

class linearRegression(iterator):
    def __next__(self):
        data = super().__next__()
        return (data['x']), (data['t'])
    
class classification(iterator):
    def __next__(self):
        data = super().__next__()
        return (data['x']), (data['label'])

class objectDetection(iterator):
    def __next__(self):
        data = super().__next__()
        return (data['img'], data['bboxs']), (data['labels'])

class SeqIterator(iterator):
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False,
                         gpu=gpu)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iteration) % self.data_size for i in
                       range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t
