#coding=utf8


import getopt
import string
import json
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
from . import LabelAsset
from . import EqualRiskAssetRatio
from . import EqualRiskAsset
from . import HighLowRiskAsset
import os
from . import DBData
from . import AllocationData
import time
from . import RiskHighLowRiskAsset
from . import ModelHighLowRisk
from . import GeneralizationPosition
from . import Const
from . import WeekFund2DayNav
from . import FixRisk
from . import DFUtil

from datetime import datetime, timedelta
from dateutil.parser import parse
from .Const import datapath

import traceback, code

@click.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help='dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2012-07-15', help='start date to calc')
@click.option('--end-date', 'enddate', help='end date to calc')
@click.option('--label-asset/--no-label-asset', default=False)
@click.pass_context
def stock(ctx, datadir, startdate, enddate, label_asset):
    '''run constant risk model
    '''
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")        

    category_stock =['date', 'largecap', 'smallcap', 'rise', 'oscillation', 'decline', 'growth', 'value']
    
    df_pool = pd.read_csv(datapath('fund_pool.csv'),  index_col=['category','date'], parse_dates=['date'], dtype={'code': str}, usecols=['category', 'date', 'code'])
    df = df_pool.loc[category_stock]
    df.index = df.index.droplevel(0)

    df_result = df.groupby(level=0, group_keys=False).apply(lambda x: x.drop_duplicates('code'))
    df_result['ratio'] = 1.0
    df_result['ratio'] = df_result['ratio'].groupby(level=0).apply(lambda x: np.round(x / len(x), 4))
    df_result.reset_index(inplace=True)
    df_result = DFUtil.pad_sum_to_one(df_result, df_result['date'], 'ratio')
    df_result['risk'] = 1.0
    df_result.set_index(['risk', 'date', 'code'], inplace=True)
    df_result.to_csv(datapath('position-pool-stock.csv'), header=False)

    print((df_result['ratio'].groupby(level=(0,1)).sum()))

                        

