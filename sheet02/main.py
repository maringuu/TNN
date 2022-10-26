import numpy as np
import matplotlib.pyplot as plt


def tanh_taylor(x):
    return x -x**3/3 + (2*x**5)/15 - (17*x**7)/315 + (62*x**9)/2835 - (1382*x**11)/155925 + (21844*x**13)/6081075 - (929569*x**15)/638512875 + (6404582*x**17)/10854718875 - (443861162*x**19)/1856156927625 + (18888466084*x**21)/194896477400625


def main():
    x = np.arange(-10, 10, 0.05)

    plt.plot(x, np.tanh(x))
    y = tanh_taylor(x)
    # For big or small values of x the taylor series is wrong
    y[x < -1.9] = -1
    y[x > 1.9] = 1
    plt.plot(x, y)
    plt.show()
