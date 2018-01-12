'''Sujet 2 part 3 script.'''

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pbinv

#Globals
f = lambda t: np.sin(t)/(1 + np.exp(t)) + np.sin(3 * t)/(1 + np.exp(-t))

NAMES = ['SigPiano.sav', 'SigDop.sav', 'Mallat.sav', 'SigTest.sav']

SNAMES = {'Mallat.sav': 'mallat',
	'SigDop.sav': 'dop',
	'SigTest.sav': 'signal',
	'SigPiano.sav': 'piano'}
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

def inv_gaussian_win(size):
    """Generate a truncated gaussian window

    Call
    ----
      window = gaussian_win(size)

    Notes
    -----
    The gaussian window is

    exp(-18 * t^2/2)

    with t in [-0.5, 0.5]
    """
    t = (np.arange(size) - (size - 1) / 2.0) / (size - 1)

    return np.exp(18 * t**2)

# Plot the signals
for i, name in enumerate(NAMES[:1]):
	# Time and frequency ranges
	N = lengths[name]
	irange = np.arange(N, dtype = 'float')
	trang = np.arange(N, dtype = 'float')/fes[name]
	frang = np.arange(N, dtype = 'float')*fes[name]/N

	# Obtain the windowed data
	matrix = pbinv.slice(signals[name], K, K)
	G = np.array([pbinv.gaussian_win(K) for i in range(N/K)])
	G = np.transpose(G)
	gaussian_matrix = G * matrix
	spectrum_raw = np.fft.fft(gaussian_matrix, axis = 0)
	spectrum = pbinv.normspec(np.abs(spectrum_raw)**2)

	# Save spectrum
	np.save(name + "spectrum_raw", spectrum_raw)

	# Rebuild slices from transformed
	rebuilt_gaussian_matrix = np.fft.ifft(np.load(name + "spectrum_raw.npy"))
	invG = np.array([inv_gaussian_win(K) for i in range(N/K)])
	invG = np.transpose(invG)
	rebuilt_matrix = invG * rebuilt_gaussian_matrix
	print("{}: {}, {}".format(name,
		np.shape(matrix), np.shape(rebuilt_matrix)))

	# Write data from slices
	rebuilt_data = [rebuilt_matrix[i][j] for i in range(K) for j in range(N/K)]
	
	# Plot Data
	plt.figure(3 * i)
	plt.clf()
	plt.plot(trang, signals[name], '+', label=name)
	plt.title('Data ' + name)
	plt.xlabel('Time (s)')
	plt.ylabel('Signal (UA)')

	#Plot rebuilt data
	plt.figure(3 * i + 1)
	plt.clf()
	plt.plot(trang, rebuilt_data, '+', label=name)
	plt.title('Rebuilt Data ' + name)
	plt.xlabel('Time (s)')
	plt.ylabel('Signal (UA)')

	# Plot Spectrum
	plt.figure(3 * i + 2)
	plt.clf()
	plt.imshow(spectrum, aspect = 'auto', origin = 'lower',
		extent = (trang[0], trang[-1], frang[0], frang[-1]))
	plt.title('Spectrogram ({}, {}) {}'.format(K, P, name))
	plt.xlabel('Time (s)')
	plt.ylabel('Frequency (Hz)')
plt.show()




