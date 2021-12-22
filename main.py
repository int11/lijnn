import optimizer
from model import *
from dataset.mnist import load_mnist
from dataset.example import *

(x, t), (x_test, t_test) = load_mnist(flatten=False)

t = oneshotencoding(t)

nn1 = nn(categorical_crossentropy())
nn1.add(Convolution(filter_size=(30, 1, 5, 5)), Relu(), Pooling(2, 2, 2))
nn1.add(Dense(100, init_sd=init.He), Relu())
nn1.add(Dense(10, init_sd=init.Xavier), Softmax())
nn1.fit(x[:5000], t[:5000], x_test[:1000], t_test[:1000], batch_size=100, epochs=300, opti=optimizer.Adam(lr=0.001))
