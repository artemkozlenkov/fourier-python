import matplotlib.pyplot as plt
import numpy as np

dt = 0.001
time = np.arange(0, 1, dt)
timeInt = len(time)
freq1 = 50
freq2 = 120


def plot_original_sines(*sines):
    fix, axs = plt.subplots(len(sines), 1)

    for i, s in enumerate(sines):
        plt.sca(axs[i])
        plt.plot(time, s)

    plt.show()
    pass


def gen_sine(time, freq):
    return np.sin(2 * np.pi * freq * time)


def gen_distorted_sig(sine):
    sine = sine + 2.5 * np.random.randn(timeInt)
    plt.plot(time, sine)
    plt.show()

    return sine


def get_psd(sig):
    psd = (sig * np.conj(sig)) / timeInt

    freqs = (1 / (dt * timeInt)) * np.arange(timeInt)
    L = np.arange(1, np.floor(timeInt / 2), dtype='int')

    plt.plot(freqs[L], np.real(psd[L]), label="PSD")
    # plt.xlim(freqs[L[0]], freqs[L[-1]])
    plt.show()

    return psd


def filter_psd(sig, psd):
    npsd = psd > 0.06

    nsig = sig * npsd

    invFft = np.fft.ifft(nsig)

    plt.plot(time, np.imag(invFft))
    plt.show()

    pass


if __name__ == '__main__':
    # step 1
    plot_original_sines(
        gen_sine(time, freq1),
        gen_sine(time, freq2),
        sum([gen_sine(time, freq1), gen_sine(time, freq2)])
    )

    # step 2
    dis_sig = gen_distorted_sig(sum([gen_sine(time, freq1), gen_sine(time, freq2)]))

    psd = get_psd(dis_sig)

    filter_psd(dis_sig, psd)

    pass
