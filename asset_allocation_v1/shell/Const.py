#coding=utf8

import os

rf = 0.025 / 52
annual_rf = 0.03


hs300_code               = '000300.SH' #沪深300
zz500_code               = '000905.SH' #中证500
largecap_code            = '399314.SZ' #巨潮大盘
smallcap_code            = '399316.SZ' #巨潮小盘
largecapgrowth_code      = '399372.SZ' #巨潮大盘成长
largecapvalue_code       = '399373.SZ' #巨潮大盘价值
smallcapgrowth_code      = '399376.SZ' #巨潮小盘成长
smallcapvalue_code       = '399377.SZ' #巨潮小盘价值


csibondindex_code         = 'H11001.CSI'  #中证全债指数
ratebondindex_code        = 'H11001.CSI'  #中证国债指数
credictbondindex_code     = 'H11073.CSI'  #中证信用债指数
convertiblebondindex_code = '000832.SH'   #中证可转债指数


sp500_code                = 'SP500.SPI'   #标普500指数
gold_code                 = 'GLNC'#黄金指数
hs_code                   = 'HSCI.HI'     #恒生指数

fund_num = 5

# 定义全局变量
version = '1.0'
verbose = False
datadir = "./tmp"

bound = {
    '120000001':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    'ERI000001':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    'ERI000002':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000014':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000052':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000053':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000054':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000055':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000056':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000057':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000058':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000059':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000060':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000061':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000062':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000063':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000064':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000065':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000066':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000067':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000068':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000069':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000070':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000071':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000072':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000073':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000074':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000075':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000076':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000077':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000078':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
    '120000079':  {'sum1': 0, 'sum2' : 0,'upper': 0.70, 'lower': 0.0},
}


def datapath(filepath) :
    return os.path.join(datadir, filepath)
