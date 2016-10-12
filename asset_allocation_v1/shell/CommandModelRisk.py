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
import ModelHighLowRisk
import Const

from datetime import datetime

@click.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2010-01-01', help=u'start date to calc')
@click.option('--label-asset/--no-label-asset', default=True)
@click.pass_context
def risk(ctx, datadir, startdate, label_asset):
    '''run constant risk model
    '''

    Const.datadir = datadir

    allocationdata = AllocationData.allocationdata()
    allocationdata.start_date = startdate
    allocationdata.all_dates()

    print "aa", allocationdata.data_start_date

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


    if label_asset:
        LabelAsset.labelasset(allocationdata)
        
    EqualRiskAssetRatio.equalriskassetratio(allocationdata.fixed_risk_asset_lookback, allocationdata.fixed_risk_asset_risk_adjust_period)
    EqualRiskAsset.equalriskasset()
    # RiskHighLowRiskAsset.highlowriskasset(allocationdata.allocation_lookback, allocationdata.allocation_adjust_period)
    ModelHighLowRisk.asset_alloc_high_low(startdate, lookback=26, adjust_period=13)



