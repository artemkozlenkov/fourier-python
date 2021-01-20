import matplotlib.pyplot as plt
import numpy as np

dt = 0.001
t = np.arange(0, 1, dt)
f = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
f_dist = f + 2.5 * np.random.randn(len(t))


def plot_init():
    # plt.plot(t, f + 2.5 * np.random.randn(len(t)), color='c', linewidth=1.5, label='Noise')
    plt.plot(t, f, color='k', linewidth=1, label='Clean')
    # plt.plot(t, f_clean, color='k', linewidth=2, label='Clean')
    plt.xlim(t[0], t[-1])
    plt.ylim(-10, 10)
    plt.legend()
    plt.show()

    pass


def make_fft():
    dt = 0.001
    t = np.arange(0, 1, dt)
    n = len(t)
    fhat = np.fft.fft(f, n)

    freq = (1 / (dt * n)) * np.arange(n)  # Create x-axis of frequencies
    L = np.arange(1, np.floor(n / 2), dtype='int')  # Only plot the first half
    fig, axs = plt.subplots(2, 1)

    indices = PSD > 100  # Find all freqs with large power
    PSDclean = PSD * indices  # Zero out all others

    for i, v in enumerate(indices):
        if v == True:
            print(PSD[v])

    fhat = indices * fhat
    ffilt = np.fft.ifft(fhat)  # Inverse FFT for filtered time signal





    fig, axs = plt.subplots(3, 1)

    plt.sca(axs[0])
    plt.plot(t, f_dist, color='c', linewidth=1, label='Noisy')
    plt.plot(t, f, color='k', linewidth=1, label='Clean')
    plt.xlim(t[0], t[-1])
    plt.legend()

    plt.sca(axs[1])
    plt.plot(freq[L], PSD[L], color='c', linewidth=2, label='Noisy')
    plt.xlim(freq[L[0]], freq[L[-1]])
    plt.legend()

    plt.sca(axs[2])
    plt.plot(t, ffilt, color='k', linewidth=0.5, label='Filtered')
    plt.xlim(t[0], t[-1])
    plt.ylim(-5, 5)
    plt.legend()
    plt.show()
    pass


def plot_original_sines():
    freq1 = 10  # frequency = 1/t
    freq2 = 20  # frequency = 1/t
    sine1 = np.sin(2 * freq1 * np.pi * t)
    sine2 = np.sin(2 * freq2 * np.pi * t)
    sum = np.sin(2 * freq1 * np.pi * t) + np.sin(2 * np.pi * freq2 * t)
    fig, axs = plt.subplots(3, 1)

    plt.sca(axs[0])
    plt.plot(t, sine1, color='black', linewidth=1, label='sine')
    plt.ylabel('First Sine', labelpad=1)
    plt.xlim(t[0], t[-1])
    plt.legend()

    plt.sca(axs[1])
    plt.plot(t, sine2, color='black', linewidth=1)
    plt.ylabel('Second Sine', labelpad=1)
    plt.xlim(t[0], t[-1])
    plt.legend()

    plt.sca(axs[2])
    plt.plot(t, sum, color='black', linewidth=1)
    plt.ylabel('Sum Sine', labelpad=1)
    plt.xlim(t[0], t[-1])
    plt.legend()

    plt.show()
    pass


def plot_matrix():
    freq1 = 10  # frequency = 1/t
    freq2 = 20  # frequency = 1/t
    sine1 = np.sin(2 * freq1 * np.pi * t)
    fftSig = np.fft.fft(sine1, len(t))

    psd = fftSig * np.conj(fftSig) / len(t)
    freqs = (1 / (dt * len(t))) * np.arange(len(t))

    l = np.arange(1, np.floor(len(t) / 2), dtype='int')

    plt.plot(freqs[l], np.real(psd[l]))
    plt.show()

    i = 0
    mtrx = []
    for a in fftSig:
        if i == 10:
            break
        mtrx.append([np.imag(a), np.real(a)])
        i += 1

    x = len(t)
    psd = fftSig * np.conj(fftSig) / x

    i = 0
    for line in fftSig:
        if (10 == i):
            break
        print(line)
        i += 1


pass

if __name__ == '__main__':
    # test_function()
    # plot_init()
    make_fft()
    # plot_wav()
    # plot_original_sines()
    # plot_matrix()
    # plot_sine()
