import time
import numpy as np
from lijnn import *
import lijnn.functions as F
import lijnn.layers as L
import os
import lijnn.functions_conv as Fc
import lijnn.accuracy as ac


class Model(Layer):
    exceptli = [core.Add, core.Mul, core.Sub, core.Div, core.Pow, F.ReLU, Fc.LocalResponseNormalization]

    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

    def layers_dict(self):
        def loop(self, parent_key=""):
            for name in self._params:
                obj = self.__dict__[name]
                key = parent_key + '/' + name if parent_key else name
                if isinstance(obj, Model):
                    loop(obj, key)
                else:
                    dict[key] = obj

        dict = {}
        loop(self)
        return dict

    def info(self, *x):
        """
        Print Model forward function information
        """
        with using_config('enable_backprop', True):
            outputs  = self(*x)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
        functions = []
        temp_funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                temp_funcs.append(f)
                seen_set.add(f)
                temp_funcs.sort(key=lambda x: x.generation)

        for i in outputs:
            add_func(i.creator)

        while temp_funcs:
            f = temp_funcs.pop()
            functions.append(f)

            for x in f.inputs:
                if x.creator != None:
                    add_func(x.creator)

        # generate text
        params_layers = self.layers_dict()

        def getLayerAndname_FromParam(Params):
            for param in Params:
                for name, obj in params_layers.items():
                    if id(param) in [id(i) for i in obj.params()]:
                        return name, obj
            return None, None

        text = []
        total_params = 0
        for i in functions[::-1]:
            if type(i) not in Model.exceptli:
                temp_li = []
                params = [input for input in i.inputs if isinstance(input, Parameter)]
                name, obj = getLayerAndname_FromParam(params)

                temp_li.append(f"{name} ({i.__class__.__name__})")

                temp = ''
                for output in i.outputs:
                    temp += f'{str(output().shape)}'
                temp_li.append(temp)

                size = obj.params_size if obj else 0
                temp_li.append(f'{size}')
                total_params += size

                text.append(temp_li)
        # print text with sort
        space = np.max([[len(e) for e in i] for i in text], axis=0)

        print(f"\n{self.__class__.__name__:=^{np.sum(space) + 15}}")
        print(f'{"Layer (type)":<{space[0] + 5}}{"Output Shape":<{space[1] + 5}}Param')
        print('=' * (np.sum(space) + 15))
        for i in text:
            for e0, e1 in zip(i, space):
                print(f'{e0:<{e1 + 5}}', end='')
            print()
        print(f"Total_params: {total_params:,}\n")

    def save_weights_epoch(self, epoch, t=None, name='default'):
        model_dir = os.path.join(utils.cache_dir, self.__class__.__name__)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        weight_dir = os.path.join(model_dir, f'{name}_{epoch}_{t}_epoch.npz' if t else f'{name}_{epoch}_epoch.npz')
        print(f'model weight save : {weight_dir}')
        self.save_weights(weight_dir)
        print('model weight save Done')

    def load_weights_epoch(self, epoch=None, ti=0, name='default', classname=None):
        model_dir = os.path.join(utils.cache_dir, classname) if classname else os.path.join(utils.cache_dir,
                                                                                            self.__class__.__name__)
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
                # ex) default_1_epoch, default_5_epoch, default_9_epoch
                temp = [i for i in name_listdir if int(i[1]) == epoch and len(i) == 3]
                # ex) default_1_120_epoch, default_1_250_epoch, default_4_90_epoch
                temp0 = [i for i in name_listdir if int(i[1]) == epoch and len(i) == 4]
                ti = 0 if temp else max([int(i[2]) for i in temp0])

            weight_dir = os.path.join(model_dir,
                                      f'{name}_{epoch}_{ti}_epoch.npz' if ti else f'{name}_{epoch}_epoch.npz')
            print(f'\n{self.__class__.__name__} Model weight load : {weight_dir}\n')
            self.load_weights(weight_dir)
        except FileNotFoundError:
            epoch = 0
            print(f"\n{self.__class__.__name__} Model weight file not found\n")

        start_epoch = int(epoch) if ti else int(epoch) + 1
        return start_epoch, ti

    def fit(self, epoch, optimizer, train_loader, test_loader=None,
            loss_function=F.softmax_cross_entropy, accuracy_function=ac.classification,
            iteration_print=False, autosave=True, autosave_time=30, name='default', gpu=True):
        
        optimizer = optimizer.setup(self)
        start_epoch, ti = self.load_weights_epoch(name=name)

        if cuda.gpu_enable and gpu:
            self.to_gpu()
            train_loader.to_gpu()
            if test_loader:
                test_loader.to_gpu()
        elif cuda.gpu_enable == False and gpu:
            print("Can't use GPU, training with CPU.")

        for i in range(start_epoch, epoch + 1):
            sum_loss, sum_acc = 0, {}
            st = time.time()
            for x, t in train_loader:
                # forward, backward, update
                y = (self(*x),)
                loss = loss_function(*y, *t)
                if accuracy_function:
                    acc = accuracy_function(*y, *t) if isinstance(t, tuple) else accuracy_function(y, t)
                else:
                    acc = 0
                    
                self.cleargrads()
                loss.backward()
                optimizer.update()


                # sum loss/acc
                sum_loss += loss.data
                for key, value in acc.items():
                    if not key in sum_acc:
                        sum_acc[key] = value.data
                    else:
                        sum_acc[key] += value.data

                if iteration_print:
                    text = f"loss : {loss.data}"
                    if accuracy_function:
                        for key, value in acc.items():
                            text += f" {key} : {value.data}"
                    print(text)

                if autosave and time.time() - st > autosave_time * 60:
                    self.save_weights_epoch(i, autosave_time + ti, name)
                    autosave_time += autosave_time

            print(f"epoch {i}")
            text = f'train loss : {sum_loss / train_loader.max_iter}'
            for key, value in sum_acc.items():
                text += f'{key} : {value / train_loader.max_iter}'
            print(text)
            self.save_weights_epoch(i, name=name)

            sum_loss, sum_acc = 0, 0
            if test_loader:
                with no_grad(), test_mode():
                    for x, t in test_loader:
                        y = (self(*x),)
                        loss = loss_function(*y, *t)
                        acc = accuracy_function(y, t).data if accuracy_function else 0
                        sum_loss += loss.data
                        sum_acc += acc
                        if iteration_print:
                            s = f"loss : {loss.data}"
                            if accuracy_function:
                                s += f"accuracy {acc}"
                            print(s)
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
