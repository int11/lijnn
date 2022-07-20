import numpy as np
import INN.functions as F


def Xavier(I, O, xp=np):
    scale = xp.sqrt(2. / (I + O))
    return scale


def He(I, xp=np):
    scale = xp.sqrt(2. / I)
    return scale


def LuCun(I, xp=np):
    scale = xp.sqrt(1. / I)
    return scale
