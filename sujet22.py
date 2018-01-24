'''Sujet 2 part 2 script.'''

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pbinv

#Globals
f = lambda t: np.sin(t)/(1 + np.exp(t)) + np.sin(3 * t)/(1 + np.exp(-t))

NAMES = ['data/SigPiano.sav',
    'data/SigDop.sav',
    'data/Mallat.sav',
    'data/SigTest.sav']
SNAMES = {'data/Mallat.sav': 'mallat',
	'data/SigDop.sav': 'dop',
	'data/SigTest.sav': 'signal',
	'data/SigPiano.sav': 'piano'}

signals = {}
fes = {}
lengths = {}
rfreqs = {}

#Copy all data in dictionaries
for name in NAMES:
	data = scipy.io.readsav(name)
	signals[name] = data[SNAMES[name]]
	fes[name] = data['fe']
	lengths[name] = len(signals[name])
	rfreqs = np.arange(lengths[name], dtype = 'float') / lengths[name]

# Slice parameters
K = 128
P = 128

def plot(K, P):
	#Plot the signals
	for i, name in enumerate(NAMES[:2]):
		# Time and frequency ranges
		N = lengths[name]
		trang = np.arange(N, dtype = 'float')/fes[name]
		frang = np.arange(N, dtype = 'float')*fes[name]/N

		# Spectrum
		matrix = pbinv.slice(signals[name], K, P)
		spectrum = np.fft.fft(matrix, axis = 0)
		spectrum = pbinv.normspec(np.abs(spectrum)**2)

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
		plt.imshow(spectrum, aspect = 'auto',
			extent = (trang[0], trang[-1], frang[0], frang[-1]))
		plt.title('Spectrogram ({}, {}) {}'.format(K, P, name))
		plt.xlabel('Time (s)')
		plt.ylabel('Frequency (Hz)')

	plt.show()
plot(K, P)
