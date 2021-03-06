import pandas as pd
import numpy as np
import numpy.fft as fft
import pywt

class TimingWt(object):

    def __init__(self, data):
        self.data = data
        self.wname = "sym4"
        self.maxlevel = 7
        #self.filter_level = [4,5,6,7]
        #self.filter_level = [4,5,7]

    def wavefilter(self, data, wavenum):
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

        dates = data.index
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
                c = pywt.wavedec(data[i,:], wname, level = self.maxlevel)

            fdata[i,:] = pywt.waverec(c, wname)

        sr = pd.Series(fdata.ravel(), index = dates)

        return sr

    def get_filtered_data(self, df, wavenum, startdate = None):
        if startdate is None:
            startdate = '2012-07-27'

        days = df[startdate:].index
        idx = []
        filtered_close = []

        for day in days:
            tmp_close = np.array(df[:day])
            tmp_filtered_close = self.wavefilter(tmp_close, wavenum)
            tmp_last_close = tmp_filtered_close[-1]
            idx.append(day)
            filtered_close.append(tmp_last_close)

        result_df = pd.DataFrame(data = filtered_close, index = idx, columns = ['wt_nav'])
        result_df['wt_inc'] = result_df.pct_change()
        result_df.index.name = 'wt_date'
        result_df = result_df.reset_index()

        return result_df


    def cal_cycle(self, data):

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

            c[0][:] = pywt.threshold(c[0], np.percentile(abs(c[0]), 100) + 1, 'soft', 0)
            c[1][:] = pywt.threshold(c[1], np.percentile(abs(c[1]), 100) + 1, 'soft', 0)
            #c[2][:] = pywt.threshold(c[2], np.percentile(abs(c[2]), 100) + 1, 'soft', 0)
            c[3][:] = pywt.threshold(c[3], np.percentile(abs(c[3]), 100) + 1, 'soft', 0)
            c[4][:] = pywt.threshold(c[4], np.percentile(abs(c[4]), 100) + 1, 'soft', 0)
            c[5][:] = pywt.threshold(c[5], np.percentile(abs(c[5]), 100) + 1, 'soft', 0)
            c[6][:] = pywt.threshold(c[6], np.percentile(abs(c[6]), 100) + 1, 'soft', 0)
            c[7][:] = pywt.threshold(c[7], np.percentile(abs(c[7]), 100) + 1, 'soft', 0)
            fdata[i,:] = pywt.waverec(c, wname)

            fdata = fdata.ravel() # If the signal is 1D, return a 1D array

        wave = fdata
        spectrum = fft.fft(wave)
        freq = fft.fftfreq(len(wave))
        order = np.argsort(abs(spectrum)[:spectrum.size/2])[::-1]
        cycle = 1/freq[order[:20]]
        max_cycle = cycle[0]/5

        return max_cycle

    def handle(self):
        yoy = np.array(self.data.close)
        window = 900
        signal = []
        ds = []
        for i in np.arange(window, len(yoy)):
            fdata = self.wavefilter(yoy[:i])
            if fdata[-1] - fdata[-2] > 0:
                signal.append(1)
            else:
                signal.append(-1)
            ds.append(self.data.index[i])
        df = pd.DataFrame(signal, index = ds, columns = ['signal'])
        #self.data = df
        #self.data.dropna(inplace = True)
        self.data = self.data.loc[df.index][['close']]
        self.data['signal'] = df['signal']
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
    data = pd.read_csv('tmp/120000001_ori_day_data.csv', index_col = 0, \
            parse_dates = True)
    tw = TimingWt(data)
    filtered_data = tw.get_filtered_data(data, 4)
    print filtered_data
