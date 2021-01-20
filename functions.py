import matplotlib.pyplot as plt
import numpy as np

dt = 0.001
t = np.arange(0, 1, dt)
freq1 = 10
freq2 = 50


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
    return  f, 'gaussian noise'


def spectralDensity():
    num = len(t)
    psd = functSumNoise() * np.conj(functSumNoise()) / num
    freqx = (1 / (dt * num)) * np.arange(num)
    L = np.arange(1, np.floor(num / 2), dtype='int')

    plt.plot(freqx[L], psd[L], color='c', linewidth=1, label='Power Spectral Density')
    # plt.xlim(freqx[L[0]], freqx[L[-1]])
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # functSumNoise()
    # functFreq10()
    # functFreq50()
    # functSumClean()
    # spectralDensity()
    singlePlot(functSumNoise())
