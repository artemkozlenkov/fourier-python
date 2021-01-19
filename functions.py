import matplotlib.pyplot as plt
import numpy as np

dt = 0.001
t = np.arange(0, 1, dt)
f = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
f_dist = f + 2.5 * np.random.randn(len(t))


def functFreq10():
    pass


def functFreq20():
    pass


if __name__ == '__main__':
    functFreq10()
    functFreq20()
