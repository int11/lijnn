import numpy as np


class nn:
    def __init__(self, x, y, actifuns, costfuns, ):
        self.h = 1e-4  # 0.0001
        self.learning_rate = 1e-2
        self.x = x
        self.y = y
        if len(self.x.shape) == 1:
            self.x = self.x[np.newaxis]
        self.lensample = len(self.x[0])
        self.w = np.zeros(len(self.x))
        self.b = np.zeros(1)

        if actifuns == "linear":
            def activation_fun():
                return np.matmul(self.w, self.x) + self.b
        elif actifuns == "sigmoid":
            def activation_fun():
                return 1 / (1 + np.exp(-(np.matmul(self.w, self.x) + self.b)))
        else:
            raise
        self.activation_fun = activation_fun

        if costfuns == "mse":
            def cost_fun():
                return np.sum((self.activation_fun() - self.y) ** 2) / self.lensample
        elif costfuns == "binary_crossentropy":
            def cost_fun():
                delta = 1e-7
                return -np.sum(self.y * np.log(self.activation_fun() + delta) + (1 - self.y) * np.log(
                    (1 - self.activation_fun()) + delta))
        elif costfuns == "categorical_crossentropy":
            raise
        else:
            raise
        self.cost_fun = cost_fun

    def __call__(self):
        self.numerical_diff(self.cost_fun, self.w)
        self.numerical_diff(self.cost_fun, self.b)

    def numerical_diff(self, f, x):
        for i in range(len(x)):
            args = x[i]
            x[i] = args + self.h
            f1 = f()
            x[i] = args - self.h
            f2 = f()
            x[i] = args
            x[i] -= self.learning_rate * (f1 - f2) / (2 * self.h)


x = np.array([[4.9,3.0,1.4,0.2],[5.8,2.6,4.0,1.2],[6.7,3.0,5.2,2.3],[5.6,2.8,4.9,2.0]])
y = np.array(["setosa","setosa","versicolor","virginica","virginica"])
nn = nn(x.T, y, "sigmoid", "binary_crossentropy")
for i in range(800000):
    nn()
    if i % 10000 == 0:
        print(nn.cost_fun(), nn.w, nn.b)
x = [[5.1,3.5,1.4,0.2]

print(nn.activation_fun())
