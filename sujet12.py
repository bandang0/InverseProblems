''' Sujet 1 part 2 script.'''

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

#Get data
data = scipy.io.readsav('data/signal.sav')
signal = data.sig
N = len(signal)

print("There are {} points.".format(N))

# Plots data
plt.figure(1)
plt.clf()
plt.plot(signal, '+')

plt.figure(2)
plt.clf()

for k in [1, 2, 4, 8, 16]:
    #Spectrum
    spectrum = np.fft.fft(signal[::k])

    # Plots spectrum
    plt.plot(np.arange(-64/k, 64/k, dtype = 'float')/128, np.abs(spectrum)**2, '+')


# Periodised
m = 12
tiled_signal = np.tile(signal, m)

plt.figure(3)
plt.clf()
plt.plot(np.abs(np.fft.fft(tiled_signal))**2, 'x')
plt.show()
