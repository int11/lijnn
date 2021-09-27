import Layer
import optimizer
from model import *
from Layer import *
from dataset.mnist import load_mnist

(x, y), (x_test, t_test) = load_mnist(normalize=True)

y = oneshotencoding(y)

nn = nn(Layer.categorical_crossentropy())
nn.add(Dense(784, 100, Layer.Relu(), initialization=initialization.He))
nn.add(Dense(100, 100, Layer.Relu(), initialization=initialization.He))
nn.add(Dense(100, 10, Layer.Softmax(), initialization=initialization.Xavier))
nn.fit(x, y, batch_size=100, epochs=10, opti=optimizer.Adam(nn, lr=0.01))
