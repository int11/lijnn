import numpy as np
import lijnn.functions as F


def Xavier(I, O, xp=np):
    scale = xp.sqrt(2. / (I + O))
    return scale


def He(I, O, xp=np):
    scale = xp.sqrt(2. / I)
    return scale


def LuCun(I, O, xp=np):
    scale = xp.sqrt(1. / I)
    return scale
