'''Sujet 2 part 1 script.'''

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

#Globals
NAMES = ['data/Mallat.sav',
    'data/SigDop.sav',
    'data/SigTest.sav',
    'data/SigPiano.sav']
SNAMES = ['mallat', 'dop', 'signal', 'piano']

signals = {}
fe = {}
lens = {}

#Copy all data in dictionaries
for name, sname in zip(NAMES, SNAMES):
    data = scipy.io.readsav(name)
    signals[name] = data[sname]
    fe[name] = data['fe']
    lens[name] = len(signals[name])

#Plot the signals
for i, name in enumerate(NAMES):
    # Time and frequency ranges
    N = lens[name]
    trang = np.arange(N, dtype = 'float')/fe[name]
    frang = np.arange(N, dtype = 'float')*fe[name]/N

    # Spectrum
    spectrum = np.fft.fft(signals[name])

    # Plot Data
    plt.figure(2 * i)
    plt.clf()
    plt.plot(trang, signals[name], '+', label=name)
    plt.title('Data ' + name)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal (UA)')

    # Plot Spectrum
    plt.figure(2 * i + 1)
    plt.clf()
    plt.plot(frang, np.abs(spectrum)**2, '+', label=name)
    plt.title('Spectrum ' + name)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectrum (UA)')

plt.show()
