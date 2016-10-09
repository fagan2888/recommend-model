#coding=utf8


import getopt
import string
import json
import os
import sys
sys.path.append('shell')
import click
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

@click.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2010-01-01', help=u'start date to calc')
@click.pass_context
def risk(ctx, datadir, startdate):
    '''run constant risk model
    '''

    Const.datadir = datadir

    allocationdata = AllocationData.allocationdata()
    allocationdata.start_date = startdate
    allocationdata.all_dates()

    print "aa", allocationdata.data_start_date
    
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
    EqualRiskAssetRatio.equalriskassetratio(allocationdata)
    EqualRiskAsset.equalriskasset(allocationdata)
    RiskHighLowRiskAsset.highlowriskasset(allocationdata)

