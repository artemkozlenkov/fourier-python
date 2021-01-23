import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dt = 0.001
t = np.arange(0, 1, dt)
l = len(t)
freq1 = 10
freq2 = 50
w = 1000


def ploty(t, f, amp, name):
    plt.plot(t, f, color='navy', linewidth=1, label='Clean', )
    plt.xlim(t[0], t[-1])
    plt.ylim(-amp, amp)
    plt.ylabel(name)
    # plt.legend()
    # plt.savefig('plot_%s.png' % name, dpi=300, bbox_inches='tight')
    # plt.show()


def multiPlot(*f):
    l = len(f)
    fig, axs = plt.subplots(l, 1)

    for i in f:
        l = l - 1
        plt.sca(axs[l])
        plt.plot(t, i[0], color='navy', linewidth=1, label=i[1], )
        plt.xlim(t[0], t[-1])
        plt.ylabel(i[1])
        plt.legend()

    plt.show()


def singlePlot(f):
    plt.plot(t, f[0], color='navy', linewidth=1, label=f[1], )
    plt.xlim(t[0], t[-1])
    plt.ylabel(f[1])
    plt.show()


def functFreq10():
    return np.sin(2 * np.pi * freq1 * t), 'sin10'


def functFreq50():
    return np.sin(2 * np.pi * freq2 * t), 'sin50'


def functSumClean():
    return functFreq10()[0] + functFreq50()[0], 'sine sum'


def functSumNoise():
    f = functSumClean()[0]
    f += 2.5 * np.random.randn(len(t))
    return f, 'gaussian noise'


def calculateFourier():
    return np.fft.fft(functSumNoise()[0], len(t))


def spectralDensity():
    f = calculateFourier()
    spectrum = f * np.conj(f) / l

    frequencies = (1 / (dt * l)) * np.arange(l)
    L = np.arange(1, np.floor(l / 2), dtype='int')

    plt.plot(frequencies[L], np.real(spectrum[L]), label='Power Spectral Density')
    plt.xlim(frequencies[L[0]], frequencies[L[-1]])
    plt.legend()
    plt.show()

    return spectrum


def filterIfourier():
    idx = spectralDensity() > 150
    f = calculateFourier() * idx
    ifourier = np.fft.fft(f, l)
    return np.real(ifourier)


def convolution():
    sin = functSumNoise()[0]
    window = np.ones(20)
    window /= sum(window)

    conv = np.convolve(sin, window, mode='valid')

    f_noise = functSumNoise()[0]

    fg, ax = plt.subplots(2, 1)

    plt.sca(ax[0])
    plt.xlim(t[0], t[-1])
    plt.plot(t, f_noise)

    plt.sca(ax[1])
    plt.plot(conv, color='red')

    # plt.plot(conv)
    plt.show()


if __name__ == '__main__':
    # functSumNoise()
    # functFreq10()
    # functFreq50()
    # functSumClean()
    # spectralDensity()
    # singlePlot(functSumNoise())
    # spectralDensity()
    # multiPlot(functSumClean(), (filterIfourier(), 'ifft') )
    convolution()
