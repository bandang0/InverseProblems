# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np


def gaussian_win(size):
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

    return np.exp(-18 * t**2)


def normspec(spectrum):
    """Spectrum of each column are normalized to 1 to help visualisation.

    Call
    ----
      normalized = normspec(spectrum)
    """
    return spectrum / np.max(spectrum, axis=0)[np.newaxis, :]


def slice(signal, length_win, step_win):
    """Slice a signal in several smaller signal stored inside a matrix.

    Call
    ----
      result = slice(signal, length_win, step_win)

    Parameters
    ----------
      - signal : signal to cut
      - length_win : size of the observation window
      - step_win : step between to succesive window

    Output
    ------
      A matrix with the smaller signals arranged in lines
    """
    size_sig = len(signal)

    nb_fen = int(1 + (size_sig - length_win) / (step_win))

    result = np.vstack(signal[idx * step_win:
                              idx * step_win + length_win]
                       for idx in range(nb_fen))

    return result.T


def kernel2d(size_x, size_y, scale):
    """2D gaussian window with a scale parameter for width of the window

    Call
    ----
      gaus = kernel2d(256, 256, 1)
    """
    x = np.reshape((np.arange(size_x) - size_x / 2) / size_x, (-1, 1))
    y = np.reshape((np.arange(size_y) - size_y / 2) / size_y, (1, -1))

    support = np.sqrt(x**2 + y**2) / scale

    return np.exp(-18 * (support**2))
