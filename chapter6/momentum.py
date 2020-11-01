import matplotlib.pyplot as plt
import numpy as np


def gd(x_start, step, g):   # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    for i in range(50):
        grad = g(x)
        x -= grad * step
        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break
    return x, passing_dot


def momentum(x_start, step, g, discount=0.9):   # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        grad = g(x)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break
    return x, passing_dot


def nesterov(x_start, step, g, discount=0.9):   # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        x_future = x - step * discount * pre_grad
        grad = g(x_future)
        pre_grad = pre_grad * 0.7 + grad
        x -= pre_grad * step

        passing_dot.append(x.copy())
        if abs(sum(grad)) < 1e-6:
            break
    return x, passing_dot


def contour(X, Y, Z, arr=None):
    plt.contour(X, Y, Z, colors='black')
    plt.plot(0, 0, marker='*')
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i + 2, 0], arr[i:i + 2, 1])


if __name__ == '__main__':
    def f(x):
        return x[0] * x[0] + 50 * x[1] * x[1]

    def g(x):
        return np.array([2 * x[0], 100 * x[1]])

    xi = np.linspace(-200, 200, 1000)
    yi = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(xi, yi)

    _, x_arr_g = gd([150, 75], 0.02, g)
    _, x_arr_m = momentum([150, 75], 0.005, g)
    _, x_arr_n = nesterov([150, 75], 0.01, g)

    Z = X * X + 50 * Y * Y
    contour(X, Y, Z, x_arr_g)
    plt.figure()
    contour(X, Y, Z, x_arr_m)
    plt.figure()
    contour(X, Y, Z, x_arr_n)
    plt.show()
