# -*- coding: utf-8 -*-
import numpy as np
# import talib

class CalTechIndic(object):
    def __init__(self, data_df):
        """
        :param data_df:输入数据
            type:DataFrame
            format:
                        close_pre    close     open     high      low        volume  \
                date
                2007-07-20    1553.08  1534.10  1553.19  1553.19  1529.20  3.745780e+09
                2007-07-23    1534.10  1541.57  1534.06  1547.23  1534.06  3.102700e+09
                2007-07-24    1541.57  1511.04  1541.57  1541.57  1508.62  4.115830e+09
                2007-07-25    1511.04  1518.09  1511.03  1524.31  1503.73  4.115830e+09
       """
        self.data = data_df #pd.read_csv(file_handle, index_col=['date'], parse_dates=['date'])
        self.data.fillna(method="ffill", inplace=True)

        self.close = np.array(self.data['close'])
        self.close_df = self.data['close']
        self.high = np.array(self.data['high'])
        self.low = np.array(self.data['low'])
        self.volume = np.array(self.data['volume'])
        self.popen = np.array(self.data['open'])
        self.tec_indic = ['macd', 'atr', 'cci', 'rsi', 'sobv', 'mtm', 'roc', 'slowkd', \
                'pct_chg', 'pvt', 'wvad', 'priceosc', 'bias', 'vma', 'vstd', 'dpo']
    def get_indic(self):
        # 计算macd
        self.macd = CalTechIndic.get_macd_hist(self.close)
        # 计算ATR
        self.atr = CalTechIndic.get_atr(self.high, self.low, self.close)
        # # 计算CCI
        self.cci = CalTechIndic.get_cci(self.high, self.low, self.close)
        # # 计算rsi
        self.rsi = CalTechIndic.get_rsi(self.close)
        # # 计算obv
        self.sobv = CalTechIndic.get_obv(self.close, self.volume)
        # # 计算MTM
        self.mtm = CalTechIndic.get_mtm(self.close)
        # 计算apo
        self.dpo = CalTechIndic.get_apo(self.close)
        # 计算roc
        self.roc = CalTechIndic.get_roc(self.close)
        # 计算slowKD
        self.slowkd = CalTechIndic.get_slowkd(self.high, self.low, self.close)
        # 计算pct_chg
        self.pct_chg = CalTechIndic.get_pct_chg(self.close_df)
        # 计算pvt
        self.pvt = CalTechIndic.get_pvt(self.close_df, self.volume)
        # 计算wvad
        self.wvad = CalTechIndic.get_wvad(self.high, self.low, self.close, \
                                        self.popen, self.volume)
        # 计算priceosc
        self.priceosc = CalTechIndic.get_priceosc(self.close)
        # 计算bias
        self.bias = CalTechIndic.get_bias(self.close_df)
        # 计算vma(与wind不一样)
        self.vma = CalTechIndic.get_vma(self.high, self.low, self.close, \
                                        self.popen)
        # 计算vstd
        self.vstd = CalTechIndic.get_vstd(self.volume)
        self.merge_data()
        return self.data
    def merge_data(self):
        """
        usage: 把技术指标数据加入原始数据
        """
        for tec_name in self.tec_indic:
            self.data[tec_name] = eval('self.' + tec_name)
    @staticmethod
    def get_macd_hist(close, fastperiod=12, slowperiod=26, signalperiod=9):
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod, slowperiod, signalperiod)
        return 2.0 * macdhist
    @staticmethod
    def get_atr(high, low, close, timeperiod=14):
        atr = talib.ATR(high, low, close, timeperiod)
        return atr
    @staticmethod
    def get_cci(high, low, close, timeperiod=14):
        cci = talib.CCI(high, low, close, timeperiod)
        return cci
    @staticmethod
    def get_rsi(close, timeperiod=12):
        rsi = talib.RSI(close, timeperiod)
        return rsi
    @staticmethod
    def get_obv(close, volume):
        obv = talib.OBV(close, volume)
        return obv
    @staticmethod
    def get_mtm(close, timeperiod=6):
        mtm = talib.MOM(close, timeperiod)
        return mtm
    @staticmethod
    def get_apo(close, fastperiod=12, slowperiod=26, matype=0):
        apo = talib.APO(close, fastperiod, slowperiod, matype)
        return apo
    @staticmethod
    def get_roc(close, timeperiod=12):
        roc = talib.ROC(close, timeperiod)
        return roc
    @staticmethod
    def get_slowkd(high, low, close,  fastk_period=9, slowk_period=3, \
                slowk_matype=0, slowd_period=3, slowd_matype=0):

        slowk, slowd = talib.STOCH(high, low, close,  fastk_period, \
                slowk_period, slowk_matype, slowd_period, slowd_matype)
        return slowd
    @staticmethod
    def get_ad(high, low, close, volume):
        ad = talib.AD(high, low, close, volume)
        return ad
    @staticmethod
    def get_pct_chg(close):
        pct_chg = close.pct_change()
        return np.array(pct_chg) * 100.0
    @staticmethod
    def get_pvt(close, volume):
        pct_chg = CalTechIndic.get_pct_chg(close)
        pct_chg[0] = 0.0
        x = pct_chg * volume
        pvt = np.add.accumulate(x)
        return pvt
    @staticmethod
    def get_wvad(high, low, close, popen, volume):
        a = close - popen
        b = high - low
        c = a / b * volume
        c = np.nan_to_num(c)
        wvad = np.add.accumulate(c)
        return wvad
    @staticmethod
    def get_priceosc(close, fastperiod=12, slowperiod=26, matype=0):
        ppo = talib.PPO(close, fastperiod, slowperiod, matype)
        return ppo
    @staticmethod
    def get_bias(close, mavalue=12):
        avg = close.rolling(window=mavalue).mean()
        bias = ((close - avg) / avg ) * 100.0
        return bias
    @staticmethod
    def get_vma(high, low, close, popen, timeperiod=6, matype=0):
        mean_price = (high + low + close + popen) / 4.0
        vma = talib.MA(mean_price, timeperiod, matype)
        return vma
    @staticmethod
    def get_vstd(volume, timeperiod=12, nbdev=1):
        vstd = talib.STDDEV(volume, timeperiod, nbdev)
        return vstd

if __name__ == "__main__":
    file_handle = "../tmp/000300_data.csv"
    obj = CalTechIndic(file_handle)
    result = obj.get_indic()
    # result.dropna(inplace=True)
    result.to_csv("000300_data_indic.csv",  encoding='utf-8')
    print(result.head(20))
