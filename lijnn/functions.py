import numpy as np
import lijnn
from lijnn import cuda, utils
from lijnn.core import Function, Variable, as_variable

# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        if isinstance(slices, Variable):
            slices = slices.data
        elif isinstance(slices, tuple):
            slices = tuple(slices[i].data if isinstance(slices[i], Variable) else
                               slices[i] for i in range(len(slices)))

        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        return GetItemGrad(self.slices, x.shape)(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = lijnn.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            import cupyx
            cupyx.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    return GetItem(slices)(x)


def expand_dims(x, axis):
    x = as_variable(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        input_shape = self.inputs[0].shape
        gy = utils.reshape_sum_backward(gy, input_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, input_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        
        """Sum elements along axes to output an array of a given shape.

        Args:
            x (ndarray): Input array.
            shape:

        Returns:
            ndarray: Output array of the shape.
        """
        ndim = len(self.shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))

        axis = tuple([i + lead for i, sx in enumerate(self.shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = lijnn.cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)


mean = average


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        # y = 1 / (1 + xp.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        return gx - (y * sumdx)


def softmax(x, axis=1):
    return Softmax(axis)(x)


class LogSoftmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)


class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx = gy * mask
        return gx


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)


# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================


class MeanSquaredError(Function):
    def forward(self, x, t):
        diff = x - t
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x, t = self.inputs
        diff = x - t
        gx = gy * diff * (2. / len(diff))
        gt = -gx
        return gx, gt


def mean_squared_error(x, t):
    return MeanSquaredError()(x, t)


class CategoricalCrossEntropy(Function):
    def forward(self, x, t):
        xp = cuda.get_array_module(x)
        N = x.shape[0]
        p = xp.clip(x, 1e-15, 1.0)
        log_p = xp.log(p[xp.arange(N), t.ravel()])
        y = -log_p.sum() / N
        return y

    def backward(self, dout=1):
        x, t = self.inputs
        return -(t / x)


def categorical_cross_entropy(y, t):
    return CategoricalCrossEntropy()(y, t)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        """simple code

        N = x.shape[0]
        p = softmax(x)
        p = clip(p, 1e-15, 1.0)  # To avoid log(0)
        log_p = log(p)
        tlog_p = log_p[np.arange(N), t.data]
        y = -1 * sum(tlog_p) / N
        """

        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        # log_p = x[np.arange(N), t.ravel()] - log_z.ravel()
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def binary_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(t)
    x = clip(x, 1e-15, 1.0)
    tlog_p = t * log(x) + (1 - t) * log(1 - x)
    y = -1 * sum(tlog_p) / N
    return y


def sigmoid_cross_entropy(x, t):
    return binary_cross_entropy(sigmoid(x), t)


def l1_loss(x, t):
    y = t - x
    return y.sum()


class SmoothL1Loss(Function):
    def __init__(self, reduction):
        assert reduction == 'sum' or reduction == 'mean'
        self.reduction = reduction

    def forward(self, x, t):
        xp = cuda.get_array_module(x)
        y = t - x
        index = xp.abs(y) <= 1
        y[index] = 0.5 * y[index] ** 2
        y[~index] = xp.abs(y[~index]) - 0.5
        y = y.sum()
        if self.reduction == 'mean':
            y /= len(t)
        return y.astype(x.dtype)

    def backward(self, gy):
        x, t = self.inputs
        diff = t - x
        return tuple(SmoothL1LossGrad(diff.data, self.reduction)(gy))


def smooth_l1_loss(x, t, reduction='mean'):
    return SmoothL1Loss(reduction)(x, t)


class SmoothL1LossGrad(Function):
    def __init__(self, diff, reduction):
        self.diff = diff
        self.reduction = reduction

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        if self.reduction == 'mean':
            gy /= len(self.diff)
        index = xp.abs(self.diff) <= 1

        gy = utils.reshape_sum_backward(gy, self.diff.shape, None, False)
        gt = xp.broadcast_to(gy, self.diff.shape).copy()

        gt[~index] = xp.sign(self.diff[~index]) * gt[~index]
        gt[index] = gt[index] * self.diff[index]  # * 2 * 0.5
        gx = -gt
        return gx, gt

    def backward(self, gx, gt):
        raise NotImplementedError


class L2Loss(Function):
    def forward(self, x, t):
        diff = t - x
        y = (diff ** 2).sum()
        return y

    def backward(self, gy):
        x, t = self.inputs
        diff = t - x
        gt = gy * diff * 2.
        gx = -gt
        return gx, gt


def l2_loss(x, t):
    return L2Loss()(x, t)


# =============================================================================
# dropout / batch_norm / embed_id
# =============================================================================

def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if lijnn.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


class BatchNorm(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        assert x.ndim == 2 or x.ndim == 4

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            # (N, C, H, W) -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = cuda.get_array_module(x)

        if lijnn.Config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            m = x.size // gamma.size
            s = m - 1. if m - 1. > 1. else 1.
            adjust = m / s  # unbiased estimation
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std
        y = gamma * xc + beta

        if x_ndim == 4:
            # (N*H*W, C) -> (N, C, H, W)
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return gx, ggamma, gbeta


def batch_nrom(x, gamma, beta, mean, var, decay=0.9, eps=2e-5):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


def embed_id(x, W):
    return W[x]


# =============================================================================
# max / min / clip / Concatenate
# =============================================================================
class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


class Concatenate(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, *x):
        xp = cuda.get_array_module(x[0])
        return xp.concatenate(x, self.axis)

    def backward(self, gx):
        xp = cuda.get_array_module(gx)
        if len(self.inputs) == 1:
            return gx
        sizes = xp.array([v.shape[self.axis] for v in self.inputs[:-1]]).cumsum()
        return tuple(split_axis(gx, sizes, self.axis))


def concatenate(x, axis=1):
    y = Concatenate(axis)(*x)
    return y


class SplitAxis(Function):
    def __init__(self, indices_or_sections, axis):
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)

        x = xp.split(x, self.indices_or_sections, self.axis)

        self._shapes = [r.shape for r in x]
        return tuple(x)

    def backward(self, gx):
        xp = cuda.get_array_module(gx)
        dtype = self.inputs[0].dtype
        grads = [xp.zeros(shape, dtype=dtype) if gy is None else gy for gy, shape in zip(gx, self._shapes)]
        return concatenate(grads, self.axis)


def split_axis(x, indices_or_sections, axis):
    return SplitAxis(indices_or_sections, axis)(x)


class Absolute(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.absolute(x)

    def backward(self, gy):
        x = self.inputs[0]
        return AbsoluteGrad(x.data)(gy)


class AbsoluteGrad(Function):
    def __init__(self, x):
        super(AbsoluteGrad, self).__init__()
        self.x = x

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        return xp.sign(self.x) * gy

    def backward(self, gy):
        return AbsoluteGrad(self.x)(gy)


def absolute(x):
    return Absolute()(x)


def abs(x):
    return Absolute()(x)


from lijnn.functions_conv import conv2d, deconv2d
from lijnn.functions_conv import im2col, col2im
from lijnn.functions_conv import max_pooling, average_pooling
from lijnn.functions_conv import local_response_normalization
from lijnn.core import add, sub, rsub, mul, div, neg, pow
