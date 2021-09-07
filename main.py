import optimizer
from model import *
from layer import *
from dataset.mnist import load_mnist

(x, y), (x_test, t_test) = load_mnist(normalize=True)

y = oneshotencoding(y)

nn = nn(cost.categorical_crossentropy)
nn.add(Dense(784, 50, activation.relu, initialization=initialization.He))
nn.add(Dense(50, 10, activation.softmax, initialization=initialization.Xavier))
nn.fit(x, y, batch_size=100, epochs=50, opti=optimizer.Adam(nn, lr=0.01))

