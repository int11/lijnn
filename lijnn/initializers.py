import numpy as np


def Xavier(I, O, dtype, xp=np):
    scale = xp.sqrt(2. / (I + O), dtype=dtype)
    return scale


def He(I, O, dtype, xp=np):
    scale = xp.sqrt(2. / I, dtype=dtype)
    return scale


def LuCun(I, O, dtype, xp=np):
    scale = xp.sqrt(1. / I, dtype=dtype)
    return scale
