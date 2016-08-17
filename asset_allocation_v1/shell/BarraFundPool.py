#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import Financial as fin
import Const
import Data
from numpy import isnan
from datetime import datetime
import pandas as pd
import AllocationData
import DBData
import FundFilter
import StockTag as ST
import FundIndicator
import DFUtil


'''
db_params = {
            "host": "rdsf4ji381o0nt6n2954.mysql.rds.aliyuncs.com",
            "port": 3306,
            "user": "jiaoyang",
            "passwd": "wgOdGq9SWruwATrVWGwi",
            "db":"asset_allocation",
            "charset": "utf8"
}
'''


dates     = DBData.all_trade_dates()
lookback  = 52

for i in range(403, len(dates)):
    end_date   = dates[i - 1]
    start_date = dates[i - lookback]
    #print start_date, end_date
    d = dates[i]

    allocationdata = AllocationData.allocationdata()
    index_df  = DBData.index_value(start_date, end_date)
    stock_df  = DBData.stock_fund_value(start_date, end_date)
    codes, indicator     = FundFilter.stockfundfilter(allocationdata, stock_df, index_df[Const.hs300_code])

    stock_df = stock_df[codes]
    stock_df.to_csv('./tmp/' + d + '.csv')
    print stock_df
    #print d, codes
