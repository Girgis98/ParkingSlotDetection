import torch
import math
from scipy.optimize import curve_fit
import numpy as np

""" Working MSE Loss Function"""


def my_loss(output, label):
    conf_flag = (label[:, 0, :, :] == 0.5)
    not_conf_flag = (label[:, 0, :, :] == -0.5)
    out = output.permute(0, 2, 3, 1)
    la = label.permute(0, 2, 3, 1)
    Loss1 = torch.mean((out[conf_flag] - la[conf_flag]) ** 2)
    Loss2 = torch.mean((out[not_conf_flag][:, 0] - la[not_conf_flag][:, 0]) ** 2)  # confidence
    return torch.abs(Loss1) + torch.abs(Loss2)


""" Trial exp Loss Function """


def decay_fun(x):
    x1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    y1 = [math.sqrt(10), 15, 30, 40, 50, 60, 80, 110, 150, 200]
    popt, _ = curve_fit(objective_power, x1, y1)
    a, b = popt
    x = torch.abs(x)
    power = torch.pow(x, b)
    y_1 = torch.mul(power, a)
    y_2 = torch.square(y_1)
    y = torch.clamp(y_2, 0, y1[-1] * y[-1])
    return y


def objective_power(x, a, b):
    """

    :param x: input x
    :param a: constant coeffienct
    :param b: power coeffiecnt
    :return: the result of fucntion
    """
    return a * pow(x, b)


def objective_exp(x, a, b):
    """

    :param x: input x
    :param a: constant coeffienct
    :param b: power of exponential coeffiecnt
    :return: the result of fucntion
    """
    return a * np.exp(x * b)


def my_loss_exp(output, label):
    conf_max = 0.5
    conf_min = -0.5
    conf_flag = (label[:, 0, :, :] == conf_max)
    not_conf_flag = (label[:, 0, :, :] == conf_min)
    out = output.permute(0, 2, 3, 1)
    la = label.permute(0, 2, 3, 1)
    Loss1_1 = torch.mean((out[conf_flag][:, 1:] - la[conf_flag][:, 1:]) ** 2)
    Loss1_2 = torch.mean(decay_fun(out[conf_flag][:, 0] - la[conf_flag][:, 0]))
    Loss2 = torch.mean(decay_fun(out[not_conf_flag][:, 0] - la[not_conf_flag][:, 0]))  # confidence
    return torch.abs(Loss1_1) + torch.abs(Loss1_2) + torch.abs(Loss2)


def decay_fn2(x):  # y = sat / (1+ initial * math.exp(-1*inflection*x[:]))
    sat = 10
    initial = 0.01
    inflection = 50

    c = sat
    a = c / initial - 1
    b = np.log(a) / inflection
    print(c, b, a)

    exp_term = torch.mul(x, -1 * b)
    exp = torch.exp(exp_term)
    exp_a = torch.mul(exp, a)
    exp_1 = torch.add(exp_a, 1)
    inv = torch.pow(exp_1, -1)
    y = torch.mul(inv, c)

    return y
