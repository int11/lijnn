def numerical_diff(dxindex, f, *x):
    h = 1e-4  # 0.0001
    x1, x2 = list(x), list(x)
    x1[dxindex] += h
    x2[dxindex] -= h
    return (f(*x1) - f(*x2)) / 2 * h


def function_temp1(*x):
    return x[0] * x[0] + x[1] * x[1]


x = [3.0, 4.0]
print(numerical_diff(0, function_temp1, *x))
print(numerical_diff(1, function_temp1, *x))
