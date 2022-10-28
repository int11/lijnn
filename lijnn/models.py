import time
from lijnn import *
import lijnn.functions as F
import lijnn.layers as L
import os


class Model(Layer):
    def __init__(self, autosave=True):
        super(Model, self).__init__()
        self.autosave = autosave

    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

    def layers_info(self, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name
            if isinstance(obj, Model):
                obj.layers_info(key)
            else:
                print(key, obj)

    def save_weights_epoch(self, epoch, t=None, name='default'):
        model_dir = os.path.join(utils.cache_dir, self.__class__.__name__)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        weight_dir = os.path.join(model_dir, f'{name}_{epoch}_{t}_epoch.npz' if t else f'{name}_{epoch}_epoch.npz')
        print(f'model weight save : {weight_dir}')
        self.save_weights(weight_dir)
        print('Done')

    def load_weights_epoch(self, epoch=None, ti=0, name='default'):
        model_dir = os.path.join(utils.cache_dir, self.__class__.__name__)
        try:
            listdir = os.listdir(model_dir)
            if not os.path.exists(model_dir):
                raise FileNotFoundError

            name_listdir = [i for i in [i.split('_') for i in listdir] if i[0] == name]

            if len(name_listdir) == 0:
                raise FileNotFoundError

            if epoch is None:
                epoch = max([int(i[1]) for i in name_listdir])
            if not ti:
                temp = [i for i in name_listdir if int(i[1]) == epoch and len(i) == 3]
                temp0 = [i for i in name_listdir if int(i[1]) == epoch and len(i) == 4]
                ti = 0 if temp else max([int(i[2]) for i in temp0])

            weight_dir = os.path.join(model_dir, f'{name}_{epoch}_{ti}_epoch.npz' if ti else f'{name}_{epoch}_epoch.npz')
            print(f'\nmodel weight load : {weight_dir}\n')
            self.load_weights(weight_dir)
        except FileNotFoundError:
            epoch = 0
            print("\nNot found any weights file, model train from scratch.\n")

        start_epoch = int(epoch) + 1
        return start_epoch, ti

    def fit(self, epoch, optimizer, train_loader, test_loader=None, name='default', iteration_print=False,
            autosave=True, autosave_time=30):
        optimizer = optimizer.setup(self)
        start_epoch, ti = self.load_weights_epoch(name=name)

        if cuda.gpu_enable:
            self.to_gpu()
            train_loader.to_gpu()
            if test_loader:
                test_loader.to_gpu()

        for i in range(start_epoch, epoch + 1):
            sum_loss, sum_acc = 0, 0
            st = time.time()
            for x, t in train_loader:
                y = self(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                self.cleargrads()
                loss.backward()
                optimizer.update()
                sum_loss += loss.data
                sum_acc += acc.data
                if iteration_print:
                    print(f"loss : {loss.data} accuracy {acc.data}")
                if autosave and time.time() - st > autosave_time * 60:
                    self.save_weights_epoch(i, autosave_time+ti, name)
                    autosave_time += autosave_time
            print(f"epoch {i + 1}")
            print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
            self.save_weights_epoch(i, name=name)

            sum_loss, sum_acc = 0, 0
            if test_loader:
                with no_grad(), test_mode():
                    for x, t in test_loader:
                        y = self(x)
                        loss = F.softmax_cross_entropy(y, t)
                        acc = F.accuracy(y, t)
                        sum_loss += loss.data
                        sum_acc += acc.data
                        if iteration_print:
                            print(f"loss : {loss.data} accuracy {acc.data}")
                print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')


class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


class SqueezeNet(Model):
    def __init__(self, pretrained=False):
        pass

    def forward(self, x):
        pass
