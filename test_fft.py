import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def print_fft(s):
    X = np.fft.fft(s)
    plt.subplot(221)
    plt.stem(abs(X))
    plt.subplot(222)
    plt.stem(np.angle(X))

    x_hat = np.fft.ifft(X)
    plt.subplot(223)
    plt.stem(np.real(x_hat))

    plt.show()

# sin signal
n = np.arange(1024.0)
s1 = np.sin(2*np.pi*n/128)
#print_fft(s1)

s2 = np.zeros(1024)
s2[0:64] = 1
print_fft(s2)