'''Sujet 3 part 1 script.'''

import numpy as np
import scipy.io
import pbinv
import matplotlib.pyplot as plt

# Globals
NAME = 'data/PicPhotographe.sav'
signalb = scipy.io.readsav(NAME).data
signal0 = scipy.io.readsav(NAME).imavrai

N = 256
frang = np.linspace(0, 1, 256, endpoint = False)

# Fourier transform of image
transformb = np.fft.fft2(signalb)

#Filter image
sigma = 5
gaussian = pbinv.kernel2d(N, N, 5)
rebuilt_image = np.fft.ifft2(gaussian * transformb)

# Plot photos
plt.figure(1)
plt.imshow(signalb,
    aspect = 'auto',
    extent = (0, N - 1, 0, N - 1))
plt.title('Data')
plt.xlabel('X (px)')
plt.ylabel('Y (px)')


plt.title('Imavrai')
plt.xlabel('X (px)')
plt.ylabel('Y (px)')

# Plot Fourier Transforms
plt.figure(4)
plt.imshow(np.abs(transformb), aspect = 'auto',
	extent = (frang[0], frang[-1], frang[0], frang[-1]))
plt.title('Fourier Data')
plt.xlabel('RFrequency')
plt.ylabel('RFrequency')

plt.show()
