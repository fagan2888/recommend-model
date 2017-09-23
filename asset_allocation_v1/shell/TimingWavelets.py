import pandas as pd
import numpy as np
import pywt

class TimingWt(object):

    def __init__(self, data):
        self.data = data
        self.wname = "sym4"
        self.maxlevel = 7
        #self.filter_level = [4,5,6,7]
        #self.filter_level = [4,5,7]

    def wavefilter(self, data):
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
        wname = self.wname

        data = np.atleast_2d(data)
        numwires, datalength = data.shape
        # Initialize the container for the filtered data
        fdata = np.empty((numwires, datalength))

        for i in range(numwires):
            # Decompose the signal
            c = pywt.wavedec(data[i,:], wname, level = self.maxlevel)
            #print c
            # Destroy the approximation coefficients
            #for j in self.filter_level:
            #    c[j][:] = 0
            # Reconstruct the signal and save it
            c[1][:] = pywt.threshold(c[1], np.percentile(abs(c[1]), 50), 'soft', 0)
            c[2][:] = pywt.threshold(c[2], np.percentile(abs(c[2]), 80), 'soft', 0)
            c[3][:] = pywt.threshold(c[3], np.percentile(abs(c[3]), 100), 'soft', 0)
            c[4][:] = pywt.threshold(c[4], np.percentile(abs(c[4]), 100), 'soft', 0)
            c[5][:] = pywt.threshold(c[5], np.percentile(abs(c[5]), 100), 'soft', 0)
            c[6][:] = pywt.threshold(c[6], np.percentile(abs(c[6]), 100), 'soft', 0)
            c[7][:] = pywt.threshold(c[7], np.percentile(abs(c[7]), 100), 'soft', 0)
            fdata[i,:] = pywt.waverec(c, wname)

        if fdata.shape[0] == 1:
            return fdata.ravel() # If the signal is 1D, return a 1D array
        else:
            return fdata # Otherwise, give back the 2D array

    def handle(self):
        yoy = np.array(self.data.close)
        window = 900
        signal = []
        for i in np.arange(window, len(yoy)+1):
            fdata = self.wavefilter(yoy[:i])
            if fdata[-1] - fdata[-2] > 0:
                signal.append(1)
            else:
                signal.append(-1)
        signal = [np.nan]*(window - 1) + signal
        self.data['signal'] = signal
        self.data = self.data.loc[:,['close', 'signal']]
        self.data.dropna(inplace = True)
        df = self.data
        df['r'] = df['close'].pct_change().fillna(0.0)
        df['_signal'] = df['signal'].shift(1).fillna(-1)
        df['_signal'][df['_signal'] < 0] = 0
        df['_signal_r'] = df['_signal'] * df['r']
        df['close_v'] = df['close'] / df['close'][0]
        df['v'] = (df['_signal_r'] + 1).cumprod()
        print df
        #self.data.to_csv('tmp/tw_sh300.csv', index_label = 'date')

if __name__ == '__main__':
    data = pd.read_csv('index_data/120000001_ori_day_data.csv', index_col = 0, \
            parse_dates = True)
    tw = TimingWt(data)
    tw.handle()
