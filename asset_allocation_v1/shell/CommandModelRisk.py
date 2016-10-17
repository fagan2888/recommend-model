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
import GeneralizationPosition
import Const
import WeekFund2DayNav
import FixRisk

from datetime import datetime, timedelta

@click.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', type=click.File(mode='w'), default='-', help=u'file used to store final result')
@click.option('--start-date', 'startdate', default='2010-01-01', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.option('--label-asset/--no-label-asset', default=True)
@click.pass_context
def risk(ctx, datadir, output, startdate, enddate, label_asset):
    '''run constant risk model
    '''

    Const.datadir = datadir

    allocationdata = AllocationData.allocationdata()
    allocationdata.start_date = startdate
    if enddate:
        allocationdata.end_date = enddate
    else:
        yesterday = (datetime.now() - timedelta(days=1)); 
        allocationdata.end_date = yesterday.strftime("%Y-%m-%d")        

    allocationdata.all_dates()

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

    print "allocation calc start date", allocationdata.start_date
    print "allocation calc end date", allocationdata.end_date
    print "history data begins at", allocationdata.data_start_date


    if label_asset:
        LabelAsset.labelasset(allocationdata)

    print "calc equal risk ratio ...."
    WeekFund2DayNav.week2day(allocationdata)
    FixRisk.fixrisk(allocationdata)
    # EqualRiskAssetRatio.equalriskassetratio(allocationdata.fixed_risk_asset_lookback, allocationdata.fixed_risk_asset_risk_adjust_period)
    print "calc equal risk ratio finished"
        

    print "calc equal risk nav...."
    EqualRiskAsset.equalriskasset()
    print "calc equal risk nav finished"

    print "calc high low risk model ..."
    # RiskHighLowRiskAsset.highlowriskasset(allocationdata.allocation_lookback, allocationdata.allocation_adjust_period)
    ModelHighLowRisk.asset_alloc_high_low(startdate, enddate, lookback=26, adjust_period=13)
    print "calc high low risk model finished"

    print "output category postion ...."
    GeneralizationPosition.portfolio_category()
    print "output category portfolio ok"

    print "output simple postion ...."
    GeneralizationPosition.portfolio_simple()
    print "output simple position ok"



