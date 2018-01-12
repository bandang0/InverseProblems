#Power spectrum
def P(sig, K, nu):
	'''Returns the power spectrum on the K first values of sig.'''
	return np.abs(sum([sig[k] * np.exp(-2 * np.pi * 1j * k * nu) for k in range(K)]))**2 / K


#Recentered power spectrum
def Xd(sig, K, n, nu):
	'''Returns the power spectrum reduced to a K-window around n.'''
	lsig = [sig[n + 1 - int(K/2) + k] for k in range(K)]
	return P(lsig, K, nu)

#time_freq = [[Xd(signals[dummy], K, n, l/lengths[dummy]) for n in range(K, lengths[dummy] - K)] for l in range(lengths[dummy])]
