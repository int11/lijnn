import numpy as np
from lijnn import Layer
import lijnn.functions as F
import lijnn.layers as L
from lijnn import utils
import os


# =============================================================================
# Model / Sequential / MLP
# =============================================================================
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

    def save_weights_epoch(self, epoch, name='default'):
        model_dir = os.path.join(utils.cache_dir, self.__class__.__name__)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        weight_dir = os.path.join(model_dir, f'{name}_{epoch}_epoch.npz')

        self.save_weights(weight_dir)
        print(f'model weight save : {weight_dir}')

    def load_weights_epoch(self, epoch=None, name='default'):
        model_dir = os.path.join(utils.cache_dir, self.__class__.__name__)
        try:
            listdir = os.listdir(model_dir)
            if not os.path.exists(model_dir):
                raise

            name_listdir = [i for i in [i.split('_') for i in listdir] if i[0] == name]

            if len(name_listdir) == 0:
                raise

            if epoch is None:
                epoch = max([int(i[1]) for i in name_listdir])
            weight_dir = os.path.join(model_dir, f'{name}_{epoch}_epoch.npz')
            print(f'\n model weight load : {weight_dir}')
            self.load_weights(weight_dir)
        except:
            epoch = 0
            print("\n Not found any weights file, model train from scratch.")

        start_epoch = int(epoch) + 1
        return start_epoch


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


# =============================================================================
# SqueezeNet
# =============================================================================
class SqueezeNet(Model):
    def __init__(self, pretrained=False):
        pass

    def forward(self, x):
        pass
