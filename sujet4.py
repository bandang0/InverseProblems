#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:34:40 2018

@author: rduque
"""

import numpy as np
import scipy.io
import scipy.signal
import scipy.misc
import matplotlib.pyplot as plt

def mplot(x, title = 'Plot'):
    plt.figure(5)
    plt.imshow(x, aspect = 'auto', origin = 'lower')
    plt.show()
    
    
signal1 = scipy.io.readsav('donnee1.sav').data
photo1 = scipy.io.readsav('donnee1.sav').photo
fftker1 = scipy.io.readsav('donnee1.sav').fftker # normalized kernel

signal2 = scipy.io.readsav('donnee2.sav').data
photo2 = scipy.io.readsav('donnee2.sav').photo
ker2 = scipy.io.readsav('donnee3.sav').ker # normalized kernel

signal3 = scipy.io.readsav('donnee3.sav').data
photo3 = scipy.io.readsav('donnee3.sav').photo
ker3 = scipy.io.readsav('donnee3.sav').ker # normalized kernel

########### FILTRAGE INVERSE ##################"#
'''
signal1_fft = np.fft.fft2(signal1)
fftker1 = np.fft.fft2(fftker1)

for i in range(250):
    for j in range(250):
        if fftker1[i][j] > 0.05:
            signal1_fft[i][j] = signal1_fft[i][j]/fftker1[i][j]
            

signal1_inv = np.fft.ifft2(signal1_fft)
'''

########### MOINDRES CARRES #####################
'''
n2 = lambda a: sum([np.abs(a[i][j])**2 for i in range(256) for j in range(256)])
conv2 = scipy.signal.convolve2d

x = signal2
alpha = 2
err_max = 1
err = 2
for i in range(128):
    err_k = conv2(x, ker2, mode = 'same') - signal2
    x = x - alpha * conv2(err_k, np.transpose(ker2), mode = 'same')
    err = n2(err_k)
    print(err)
mplot(x, 'MC Build')
mplot(photo2, 'Photo2')
'''
############# MOINDRES CARRES ETENDUS ##############

n2 = lambda a: sum([np.abs(a[i][j])**2 for i in range(256) for j in range(256)])
conv2 = lambda a, b: scipy.signal.convolve2d(a, b, mode = 'same')

D = np.array([[0, -1, 1]])
mu = 0.005
x = signal2
alpha = 2
err_max = 1
err = 2
for i in range(256):
    err_k = conv2(x, ker2) - signal2
    x = x - alpha * (conv2(err_k, np.transpose(ker2)) + mu * conv2(conv2(x, D), np.transpose(D)))
    print(n2(x - photo2))
    scipy.misc.imsave('step' + str(i) + '.png', x)
mplot(x, 'MCE Build')


########### FILTRE DE WIENER #####################
'''
mu = 0.1

D = np.zeros((250, 250))
D[0, 0] = 2
D[1, 0] = -1
D[0, 1] = -1
fftker1 = np.fft.fft2(fftker1)
image_debruite = np.fft.ifft2(np.fft.fft2(signal1)*np.conjugate(fftker1)/(np.abs(fftker1)**2 + mu*np.abs(np.fft.fft2(D))**2))

mplot(np.real(image_debruite))
'''
########### PLOTS ################################
'''
plt.figure(4)
plt.imshow(signal1, aspect = 'auto', origin = 'lower')
plt.title('Data1')

plt.figure(5)
plt.imshow(np.real(signal1_inv), aspect = 'auto', origin = 'lower')
plt.title('Data2')
'''
