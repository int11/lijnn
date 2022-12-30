import example.CNN
import lijnn
import cv2 as cv
import numpy as np
from lijnn import cuda

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
img_path = lijnn.utils.get_file(url)
img = cv.imread(img_path)
img = img.transpose(2, 0, 1)[::-1]
print(img.shape)
model = example.CNN.VGG16(imagenet_pretrained=True)
if cuda.gpu_enable:
    model.to_gpu()
    img = cuda.as_cupy(img)
xp = cuda.get_array_module(img)

model.info((1, 3, 224, 224))
pre = model.predict(img, [103.939, 116.779, 123.68], 1)
print(lijnn.datasets.ImageNet.labels()[np.argmax(pre)])