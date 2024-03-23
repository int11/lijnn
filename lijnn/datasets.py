import os
import gzip
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lijnn.utils import get_file, cache_dir
from lijnn.transforms import *
import xml.etree.ElementTree as ET
import cv2 as cv
from PIL import Image
from abc import ABC, abstractmethod

class Dataset(ABC):
    def __init__(self, train=True):
        self.train = train
        self.transform = {}
    
    def add_transform(self, key, transform):
        self.transform[key] = transform

    @abstractmethod
    def getitem(self, index):
        pass 
        
    def __getitem__(self, index):
        data = self.getitem(index)
        for key in self.transform.keys():
            if key in data:
                data[key] = self.transform[key](data[key])
        return data

    def __len__(self):
        raise NotImplementedError


class Spiral(Dataset):
    def __init__(self, train=True, x_transform=None, t_transform=None):
        super().__init__(train, x_transform, t_transform)
        seed = 1984 if self.train else 2020
        np.random.seed(seed=seed)

        num_data, num_class, input_dim = 100, 3, 2
        data_size = num_class * num_data
        x = np.zeros((data_size, input_dim), dtype=np.float32)
        t = np.zeros(data_size, dtype=np.int)

        for j in range(num_class):
            for i in range(num_data):
                rate = i / num_data
                radius = 1.0 * rate
                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
                ix = num_data * j + i
                x[ix] = np.array([radius * np.sin(theta),
                                  radius * np.cos(theta)]).flatten()
                t[ix] = j
        # Shuffle
        indices = np.random.permutation(num_data * num_class)
        self.data, self.labels = x[indices], t[indices]

    def __getitem__(self, index):
        assert np.isscalar(index)
        return self.x_transform(self.data[index]), self.t_transform(self.labels[index])

    def __len__(self):
        return len(self.data)


class MNIST(Dataset):
    """
    mean = [33.31842145],
    std = [78.56748998]
    """

    def __init__(self, train=True, x_transform=None, t_transform=None):
        super().__init__(train, x_transform, t_transform)
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        with gzip.open(data_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        self.data = data.reshape(-1, 1, 28, 28)

        with gzip.open(label_path, 'rb') as f:
            self.labels = np.frombuffer(f.read(), np.uint8, offset=8)

    def __getitem__(self, index):
        assert np.isscalar(index)
        return self.x_transform(self.data[index]), self.t_transform(self.labels[index])

    def __len__(self):
        return len(self.data)

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        cv.imshow('asd', img)
        cv.waitKey(0)

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


class CIFAR10(Dataset):
    """
    mean = [125.30691805, 122.95039414, 113.86538318],
    std = [62.99321928, 62.08870764, 66.70489964]
    """

    def __init__(self, train=True, x_transform=None, t_transform=None):
        super().__init__(train, x_transform, t_transform)

        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        loaded = load_cache_npz(url, self.train)
        if loaded is not None:
            self.data, self.labels = loaded[0], loaded[1]
            return

        filepath = get_file(url)
        file = tarfile.open(filepath, 'r:gz')
        if self.train:
            self.data = np.empty((50000, 3 * 32 * 32), dtype=np.uint8)
            self.labels = np.empty((50000), dtype=np.uint8)
            for i in range(5):
                data_dict = pickle.load(file.extractfile(file.getmember(f'cifar-10-batches-py/data_batch_{i + 1}')),
                                        encoding='bytes')
                self.data[i * 10000:(i + 1) * 10000] = data_dict[b'data']
                self.labels[i * 10000:(i + 1) * 10000] = np.array(data_dict[b'labels'])
        else:
            data_dict = pickle.load(file.extractfile(file.getmember('cifar-10-batches-py/test_batch')),
                                    encoding='bytes')
            self.data = data_dict[b'data']
            self.labels = np.array(data_dict[b'labels'])

        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz({'data': self.data, 'label': self.labels}, url, self.train)

    def __getitem__(self, index):
        assert np.isscalar(index)
        return self.x_transform(self.data[index]), self.t_transform(self.labels[index])

    def __len__(self):
        return len(self.data)

    def show(self, row=10, col=10):
        H, W = 32, 32
        img = np.zeros((H * row, W * col, 3))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].transpose(1, 2, 0)
        img = img[:, :, ::-1].astype(np.uint8)
        cv.imshow('asd', img)
        cv.waitKey(0)

    @staticmethod
    def labels():
        return {0: 'ariplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                8: 'ship', 9: 'truck'}


class CIFAR100(CIFAR10):
    """
    mean = [129.30416561, 124.0699627,  112.43405006],
    std = [68.1702429,  65.39180804, 70.41837019]
    """

    def __init__(self, train=True, x_transform=None, t_transform=None, label_type='fine'):
        assert label_type in ['fine', 'coarse']
        self.label_type = label_type
        super().__init__(train, x_transform, t_transform)

        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        loaded = load_cache_npz(url, self.train)
        if loaded is not None:
            self.data, self.labels = loaded[0], loaded[1]
            return

        filepath = get_file(url)
        file = tarfile.open(filepath, 'r:gz')
        if self.train:
            data_dict = pickle.load(file.extractfile(file.getmember('cifar-100-python/train')), encoding='bytes')
        else:
            data_dict = pickle.load(file.extractfile(file.getmember('cifar-100-python/test')), encoding='bytes')

        self.data = data_dict[b'data']
        if self.label_type == 'fine':
            self.labels = np.array(data_dict[b'fine_labels'])
        elif self.label_type == 'coarse':
            self.labels = np.array(data_dict[b'coarse_labels'])
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz({'data': self.data, 'label': self.labels}, url, self.train)

    def __getitem__(self, index):
        assert np.isscalar(index)
        return self.x_transform(self.data[index]), self.t_transform(self.labels[index])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def labels(label_type='fine'):
        coarse_labels = dict(enumerate(['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                                        'household electrical device', 'household furniture', 'insects',
                                        'large carnivores', 'large man-made outdoor things',
                                        'large natural outdoor scenes', 'large omnivores and herbivores',
                                        'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles',
                                        'small mammals', 'trees', 'vehicles 1', 'vehicles 2']))
        fine_labels = []
        return fine_labels if label_type == 'fine' else coarse_labels


class VOCDetection(Dataset):
    DATASET_YEAR_DICT = {
        "2012": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "2011": "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
        "2010": "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
        "2009": "http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
        "2008": "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
        "2007": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"}
    

    lables = {label: index for index, label in enumerate(
        ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
         "train", "tvmonitor"])}
    
    def __init__(self, train=True, year=2007):
        assert 2007 <= year <= 2012
        
        super().__init__(train, None, None)

        self.year = str(year)
        url = self.DATASET_YEAR_DICT[str(self.year)]
        filepath = get_file(url)
        
        self.file = tarfile.open(filepath, 'r')
        with tarfile.open(filepath, 'r') as tar:
            tar.extractall(cache_dir)

        self.dir = os.path.join(cache_dir, "VOCdevkit/VOC2007")
        self.scan()

    def scan(self, imgdirName="JPEGImages"):
        self.imgdir = os.path.join(self.dir, imgdirName)

        filenames = os.listdir(self.imgdir)
        filenames = [os.path.splitext(filename)[0] for filename in filenames]
        self.nameindex = sorted(filenames)
        
    def getAnnotations(self, index):
        xml = ET.parse(os.path.join(self.dir, "Annotations", self.nameindex[index] + ".xml"))

        bboxes, label = [], []
        for i in xml.iter(tag="object"):
            budbox = i.find("bndbox")
            bboxes.append([int(budbox.find(i).text) for i in ['xmin', 'ymin', 'xmax', 'ymax']])
            label.append(self.lables[i.find("name").text])
        return {"label":np.array(label), "bboxes":np.array(bboxes)}

    def getImg(self, index):
        imageDir = os.path.join(self.imgdir, self.nameindex[index] + ".jpg")
        if not os.path.exists(imageDir):
            imageDir = os.path.join(self.imgdir, self.nameindex[index] + ".png")
        img = np.array(Image.open(imageDir))
        # height width channel RGB -> channel height width RGB
        img = img.transpose(2, 0, 1)
        return img
     
    def __getitem__(self, index):
        assert np.isscalar(index)
        annotations = self.getAnnotations(index)
        data = {"img":self.getImg(index), **annotations}
        return data

    def __len__(self):
        return len(self.nameindex)

class VOCclassfication(VOCDetection):
    def __init__(self, train=True, year=2007):
        super(VOCclassfication, self).__init__(train, year)

        imgdirName = "classficationImages"
        imgdir = os.path.join(self.dir, imgdirName)
        if not os.path.exists(imgdir):
            os.makedirs(imgdir, exist_ok=True)
            label = []

            for i in range(super().__len__()):
                data = super().__getitem__(i)
                label.extend(list(data["label"]))

                for e in range(len(data["bboxes"])):
                    bbox = data["bboxes"][e]
                    img = data["img"][:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    imgpath = os.path.join(imgdir, self.nameindex[i] + f"{e:02}" + ".png")
                    cv.imwrite(imgpath, img[::-1].transpose(1, 2, 0))
            np.savetxt(os.path.join(imgdir,'label.txt'), np.array(label), fmt='%d')

        self.label = np.loadtxt(os.path.join(imgdir,'label.txt'), dtype=np.int32)
        self.scan(imgdirName)

    def __getitem__(self, index):
        img = self.getImg(index)
        label = self.label[index]
        return {"img":img, "label":label}

    def __len__(self):
        return len(self.count)


class ImageNet(Dataset):
    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = get_file(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels


class SinCurve(Dataset):
    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]


class Shakespear(Dataset):
    def prepare(self):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        file_name = 'shakespear.txt'
        path = get_file(url, file_name)
        with open(path, 'r') as f:
            data = f.read()
        chars = list(data)

        char_to_id = {}
        id_to_char = {}
        for word in data:
            if word not in char_to_id:
                new_id = len(char_to_id)
                char_to_id[word] = new_id
                id_to_char[new_id] = word

        indices = np.array([char_to_id[c] for c in chars])
        self.data = indices[:-1]
        self.label = indices[1:]
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char


# =============================================================================
# Utils
# =============================================================================
def load_cache_npz(filename, train=False):
    filename = filename[filename.rfind('/') + 1:]
    prefix = '.train.npz' if train else '.test.npz'
    filepath = os.path.join(cache_dir, filename + prefix)
    if not os.path.exists(filepath):
        return None

    loaded = np.load(filepath)
    return [loaded[i] for i in loaded]


def save_cache_npz(kwargs, filename, train=False):
    filename = filename[filename.rfind('/') + 1:]
    prefix = '.train.npz' if train else '.test.npz'
    filepath = os.path.join(cache_dir, filename + prefix)
    if os.path.exists(filepath):
        return

    print("Saving: " + filename + prefix)
    try:
        np.savez_compressed(filepath, **kwargs)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise
    print(" Done")
    return filepath
