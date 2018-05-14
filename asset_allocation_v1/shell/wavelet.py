#coding=utf8


import pandas as pd
import numpy as np
import numpy.fft as fft
import pywt


class Wavelet(object):

    @staticmethod
    def wavefilter(data, wavenum, wname = 'sym4', maxlevel = 7):
        """
        This function requires that the NumPy and PyWavelet packages
        are installed. They are freely available at:

        NumPy - http://numpy.scipy.org/
        PyWavelets - http://www.pybytes.com/pywavelets/#download

        Filter a multi-channel signal using wavelet filtering.

        data     - n x m array, n=number of channels in the signal,
                    m=number of samples in the signal
        maxlevel - the level of decomposition to perform on the data. This integer
                    implicitly defines the cutoff frequency of the filter.
                    Specifically, cutoff frequency = samplingrate/(2^(maxlevel+1))
        """

        if len(data)%2 == 1:
            data = data[1:]
        # We will use the Daubechies(4) wavelet

        dates = data.index
        data = np.atleast_2d(data)
        numwires, datalength = data.shape
        # Initialize the container for the filtered data
        fdata = np.empty((numwires, datalength))

        for i in range(numwires):
            # Decompose the signal
            c = pywt.wavedec(data[i,:], wname, level = maxlevel)
            #print c
            # Destroy the approximation coefficients
            #for j in self.filter_level:
            #    c[j][:] = 0
            # Reconstruct the signal and save it
            c[1][:] = pywt.threshold(c[1], np.percentile(abs(c[1]), 50), 'soft', 0)
            c[2][:] = pywt.threshold(c[2], np.percentile(abs(c[2]), 80), 'soft', 0)
            c[3][:] = pywt.threshold(c[3], np.percentile(abs(c[3]), 80), 'soft', 0)
            c[4][:] = pywt.threshold(c[4], np.percentile(abs(c[4]), 80), 'soft', 0)
            c[5][:] = pywt.threshold(c[5], np.percentile(abs(c[5]), 80), 'soft', 0)
            c[6][:] = pywt.threshold(c[6], np.percentile(abs(c[6]), 80), 'soft', 0)
            c[7][:] = pywt.threshold(c[7], np.percentile(abs(c[7]), 80), 'soft', 0)

            if wavenum <= 6:
                c[7][:] = pywt.threshold(c[7], np.percentile(abs(c[7]), 100) + 1, 'soft', 0)
            if wavenum <= 5:
                c[6][:] = pywt.threshold(c[6], np.percentile(abs(c[6]), 100) + 1, 'soft', 0)
            if wavenum <= 4:
                c[5][:] = pywt.threshold(c[5], np.percentile(abs(c[5]), 100) + 1, 'soft', 0)
            if wavenum <= 3:
                c[4][:] = pywt.threshold(c[4], np.percentile(abs(c[4]), 100) + 1, 'soft', 0)
            if wavenum <= 2:
                c[3][:] = pywt.threshold(c[3], np.percentile(abs(c[3]), 100) + 1, 'soft', 0)
            if wavenum <= 1:
                c[2][:] = pywt.threshold(c[2], np.percentile(abs(c[2]), 100) + 1, 'soft', 0)
            if wavenum == 0:
                c = pywt.wavedec(data[i,:], wname, level = maxlevel)

            fdata[i,:] = pywt.waverec(c, wname)

        sr = pd.Series(fdata.ravel(), index = dates)

        return sr
