import os
import subprocess
import urllib.request
import numpy as np
from lijnn import as_variable
from lijnn.cuda import as_numpy
from lijnn import Variable
from lijnn import cuda
import cv2 as cv
import sys
import time

"""
if use colab, os.path.expanduser() function return "/root"

drive mount directory have to be "/content/drive" 
ex) drive.mount("/content/drive")
"""

try:
    if 'root' in os.path.expanduser('~'):
        cache_dir = '/content/drive/MyDrive/.lijnn'
    else:
        cache_dir = os.path.join(os.path.expanduser('~'), '.lijnn')
except:
    print("No cache dir found to store weights.")
    

# =============================================================================
# Visualize for computational graph
# =============================================================================
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f):
    # for function
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    # for edge
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:  # y is weakref
        ret += dot_edge.format(id(f), id(y()))
    return ret


def get_dot_graph(output, verbose=True):
    """Generates a graphviz DOT text of a computational graph.

    Build a graph of functions and variables backward-reachable from the
    output. To visualize a graphviz DOT text, you need the dot binary from the
    graphviz package (www.graphviz.org).

    Args:
        output (dezero.Variable): Output variable from which the graph is
            constructed.
        verbose (bool): If True the dot graph contains additional information
            such as shapes and dtypes.

    Returns:
        str: A graphviz DOT text consisting of nodes and edges that are
            backward-reachable from the output
    """
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    graph_path = os.path.join(cache_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]  # Extension(e.g. png, pdf)
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    # =============================================================================
    # Utility functions for numpy (numpy magic)
    # =============================================================================


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for lijnn.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def logsumexp(x, axis=1):
    xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape


# =============================================================================
# Gradient check
# =============================================================================
def gradient_check(f, x, *args, rtol=1e-4, atol=1e-5, **kwargs):
    """Test backward procedure of a given function.

    This automatically checks the backward-process of a given function. For
    checking the correctness, this function compares gradients by
    backprop and ones by numerical derivation. If the result is within a
    tolerance this function return True, otherwise False.

    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A traget `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.

    Returns:
        bool: Return True if the result is within a tolerance, otherwise False.
    """
    x = as_variable(x)
    x.data = x.data.astype(np.float64)

    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    res = array_allclose(num_grad, bp_grad, atol=atol, rtol=rtol)

    if not res:
        print('')
        print('========== FAILED (Gradient Check) ==========')
        print('Numerical Grad')
        print(' shape: {}'.format(num_grad.shape))
        val = str(num_grad.flatten()[:10])
        print(' values: {} ...'.format(val[1:-1]))
        print('Backprop Grad')
        print(' shape: {}'.format(bp_grad.shape))
        val = str(bp_grad.flatten()[:10])
        print(' values: {} ...'.format(val[1:-1]))
    return res


def numerical_grad(f, x, *args, **kwargs):
    """Computes numerical gradient by finite differences.

    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A target `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.

    Returns:
        `ndarray`: Gradient.
    """
    eps = 1e-4

    x = x.data if isinstance(x, Variable) else x
    xp = cuda.get_array_module(x)
    if xp is not np:
        np_x = cuda.as_numpy(x)
    else:
        np_x = x
    grad = xp.zeros_like(x)

    it = np.nditer(np_x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx].copy()

        x[idx] = tmp_val + eps
        y1 = f(x, *args, **kwargs)  # f(x+h)
        if isinstance(y1, Variable):
            y1 = y1.data
        y1 = y1.copy()

        x[idx] = tmp_val - eps
        y2 = f(x, *args, **kwargs)  # f(x-h)
        if isinstance(y2, Variable):
            y2 = y2.data
        y2 = y2.copy()

        diff = (y1 - y2).sum()
        grad[idx] = diff / (2 * eps)

        x[idx] = tmp_val
        it.iternext()
    return grad


def array_equal(a, b):
    """True if two arrays have the same shape and elements, False otherwise.

    Args:
        a, b (numpy.ndarray or cupy.ndarray or lijnn.Variable): input arrays
            to compare

    Returns:
        bool: True if the two arrays are equal.
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    a, b = cuda.as_numpy(a), cuda.as_numpy(b)
    return np.array_equal(a, b)


def array_allclose(a, b, rtol=1e-4, atol=1e-5):
    """Returns True if two arrays(or variables) are element-wise equal within a
    tolerance.

    Args:
        a, b (numpy.ndarray or cupy.ndarray or lijnn.Variable): input arrays
            to compare
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.

    Returns:
        bool: True if the two arrays are equal within the given tolerance,
            False otherwise.
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    a, b = cuda.as_numpy(a), cuda.as_numpy(b)
    return np.allclose(a, b, atol=atol, rtol=rtol)


# =============================================================================
# download function
# =============================================================================

def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/lijnn`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print(f"Downloading: {file_name} to {file_path}")

    def show_progress(block_num, block_size, total_size):
        bar_template = "\r[{}] {:.2f}%"

        downloaded = block_num * block_size
        p = downloaded / total_size * 100
        i = int(downloaded / total_size * 30)
        if p >= 100.0: p = 100.0
        if i >= 30: i = 30
        bar = "#" * i + "." * (30 - i)
        print(bar_template.format(bar, p), end='')

    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


# =============================================================================
# others
# =============================================================================
def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


def printoptions(precision=6, threshold=np.inf, suppress=True):
    np.set_printoptions(precision=precision, threshold=threshold, suppress=suppress)


def IOU(bbox1, bbox2):
    bbox1, bbox2 = as_numpy(bbox1), as_numpy(bbox2)
    xp = cuda.get_array_module(bbox1)
    p0 = xp.maximum(bbox1[:2], bbox2[:2])
    p1 = xp.minimum(bbox1[2:], bbox2[2:])
    if p1[0] < p0[0] or p1[1] < p0[1]:
        return 0.0
    Overlap = (p0[0] - p1[0]) * (p0[1] - p1[1])

    bb1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bb2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    Union = bb1_area + bb2_area - Overlap

    return float(Overlap / Union)


def NMS(bboxs, probs, iou_threshold=0.5):
    # xp = cuda.get_array_module(bboxs)
    # scores = xp.max(probs, axis=1)
    # order = scores.argsort()[::-1]
    #
    # keep = []
    # while order.size > 0:
    #     target, order = order[0], order[1:]
    #     keep.append(target)
    #     iou = xp.array([IOU(bboxs[target], i) for i in bboxs[order]])
    #     order = order[xp.where(iou <= iou_threshold)]
    #
    # return keep

    xp = cuda.get_array_module(bboxs)
    scores = xp.max(probs, axis=1)

    areas = (bboxs[:, 2] - bboxs[:, 0]) * (bboxs[:, 3] - bboxs[:, 1])
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        target, order = order[0], order[1:]
        keep.append(target)

        xy1 = xp.maximum(bboxs[target][:2], bboxs[order][:, :2])
        xy2 = xp.minimum(bboxs[target][2:], bboxs[order][:, 2:])

        Overlap = (xy1[:, 0] - xy2[:, 0]) * (xy1[:, 1] - xy2[:, 1])
        Union = areas[target] + areas[order] - Overlap
        iou = Overlap / Union
        iou[xp.bitwise_or(iou < 0, iou > 1)] = 0

        order = order[xp.where(iou <= iou_threshold)]

    return keep


class Timer:
    def __init__(self, func_name: str = 'this func'):
        self.func_name: str = func_name
        self.time_start: float = 0.0

    def start(self):
        sys.stdout.flush()
        self.time_start = time.perf_counter()

    def end(self):
        time_end = time.perf_counter()
        interval = time_end - self.time_start
        return interval
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        print(f'{self.func_name}: {self.end()} sec')

    def function_speed_check(self, iterint, func, *args ,**kwargs):
        func(*args)

        time_stack = np.zeros(iterint)
        for i in range(iterint):
            self.start()
            func(*args, **kwargs)
            interval = self.end()
            time_stack[i] = interval
        print(f'{func.__name__}: {np.average(time_stack)}')
        return time_stack