# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import talib

class CalTechIndic(object):
    def __init__(self, file_handle):
        self.data = pd.read_csv(file_handle, index_col=['date'], parse_dates=['date'])
        close = np.array(self.data['close'])
        high = np.array(self.data['high'])
        low = np.array(self.data['low'])
        volume = np.array(self.data['volume'])
        print self.data.index
        # 计算macd
        # macd_hist = CalTechIndic.get_macd(close)
        # 计算ATR
        # atr = CalTechIndic.get_atr(high, low, close)
        # # 计算CCI
        # cci = CalTechIndic.get_cci(high, low, close)
        # # 计算rsi
        # rsi = CalTechIndic.get_rsi(close)
        # # 计算obv
        # obv = CalTechIndic.get_obv(close, volume)
        # # 计算MTM
        # mtm = CalTechIndic.get_mtm(close)
        # 计算apo
        # apo = CalTechIndic.get_apo(close)
        # 计算roc
        # roc = CalTechIndic.get_roc(close, 12)
        # 计算slowKD
        # slowkd = CalTechIndic.get_slowkd(high, low, close)
        ad = CalTechIndic.get_ad(high, low, close, volume)
        print ad[-30:-5]
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
    def get_roc(close, timeperiod=10):
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

if __name__ == "__main__":
    file_handle = "../tmp/000300_data.csv"
    obj = CalTechIndic(file_handle)
