import optimizer
from model import *
from function import *
from layer import *
from dataset.mnist import load_mnist
(x_train, y_train), (x_test, t_test) = load_mnist(normalize=True)




nn = nn(cost.categorical_crossentropy)
nn.add(Dense(784, 100, activation.relu, initialization=initialization.Xavier))
nn.add(Dense(100, 100, activation.relu, initialization=initialization.Xavier))
nn.add(Dense(100, 10, activation.softmax, initialization=initialization.He))
nn.fit(x_train, y_train,batch_size=100, epochs=301, opti=optimizer.Adam(nn, lr=0.01))

print(nn.predict(x_test))
