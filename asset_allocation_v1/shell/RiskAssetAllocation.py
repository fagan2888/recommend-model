#coding=utf8

import getopt
import string
import json
import os
import sys
sys.path.append('shell')
import pandas as pd
import LabelAsset
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset
import os
import DB
import AllocationData
import time
import RiskHighLowRiskAsset
import Const

from datetime import datetime

def usage():
    print

def risk_asset_allocation():


    allocationdata = AllocationData.allocationdata()
    allocationdata.start_date                            = '2010-01-15'
    allocationdata.all_dates()

    #allocationdata.all_dates()

    #allocationdata.start_date                            = args.get('start_date')
    #allocationdata.start_date                            = '2010-01-01'
    #allocationdata.start_date                            = '2016-01-01'

    allocationdata.fund_measure_lookback                 = 52
    allocationdata.fund_measure_adjust_period            = 26
    allocationdata.jensen_ratio                          = 0.5
    allocationdata.sortino_ratio                         = 0.5
    allocationdata.ppw_ratio                             = 0.5
    allocationdata.stability_ratio                       = 0.5
    allocationdata.fixed_risk_asset_lookback             = 52
    allocationdata.fixed_risk_asset_risk_adjust_period   = 5
    allocationdata.allocation_lookback                   = 26
    allocationdata.allocation_adjust_period              = 13



    LabelAsset.labelasset(allocationdata)
    EqualRiskAssetRatio.equalriskassetratio(allocationdata.fixed_risk_asset_lookback, allocationdata.fixed_risk_asset_risk_adjust_period)
    EqualRiskAsset.equalriskasset()
    RiskHighLowRiskAsset.highlowriskasset(allocationdata.allocation_lookback, allocationdata.allocation_adjust_period)
    #HighLowRiskAsset.highlowriskasset(allocationdata)


    #DB.fund_measure(allocationdata)
    #DB.label_asset(allocationdata)
    #DB.asset_allocation(allocationdata)
    #DB.riskhighlowriskasset(allocationdata)



if __name__ == '__main__':

    #
    # 处理命令行参数
    #
    try:
        longopts = ['datadir=', 'verbose', 'help', ]
        options, remainder = getopt.gnu_getopt(sys.argv[1:], 'hvd:', longopts)
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in options:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        elif opt in ('-d', '--datadir'):
            Const.datadir = arg
        elif opt in ('-v', '--verbose'):
            verbose = True
        elif opt == '--version':
            version = arg

    #
    # 确认数据目录存在
    #
    if not os.path.exists(Const.datadir):
        os.mkdir(Const.datadir)
    else:
        if not os.path.isdir(Const.datadir):
            print "path [%s] not dir" % Const.datadir
            sys.exit(-1)

    print Const.datadir
    #
    # 运行资产配置程序
    #
    risk_asset_allocation()
    try:

        pass
    except Exception as e:
        print "Unexpected error:", sys.exc_info()[0]
