'''Sujet 1 part 1 script.'''

import numpy as np
import matplotlib.pyplot as plt

# Globals
N = 512
T0 = 10
Nu0 = 0.1
Te = 1
Nue = 1

# Data
k = np.arange(N)
xk = np.sin(2 * np.pi * k / 10)

# Fourier transforms
xf512 = np.fft.fft(xk)
xk510 = np.fft.fft(xk[:510])

plt.figure(1)
plt.clf()
plt.plot(k, xk, '.')

plt.figure(2)
plt.clf()
plt.plot(k / (N * Te), np.abs(xf512)**2, '.')
plt.show()


