#coding=utf8


import getopt
import string
import json
import os
import sys
import logging
sys.path.append('shell')
import click
import config
import pandas as pd
import numpy as np
import LabelAsset
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset
import os
import DBData
import AllocationData
import time
import RiskHighLowRiskAsset
import ModelHighLowRisk
import GeneralizationPosition
import Const
import WeekFund2DayNav
import FixRisk
import DFUtil
import LabelAsset
import Financial as fin

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund
from db.asset_stock_factor import *

import traceback, code


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def sf(ctx):
    '''multi factor
    '''
    pass


@sf.command()
@click.option('--filepath', 'optfilepath', help=u'stock factor infos')
@click.pass_context
def insert_factor_info(ctx, optfilepath):
    '''insert factor info
    '''
    sf_df = pd.read_csv(optfilepath.strip())
    sf_df.factor_formula = sf_df.factor_formula.where((pd.notnull(sf_df.factor_formula)), None)

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for i in range(0, len(sf_df)):
        record = sf_df.iloc[i]

        factor = stock_factor()
        factor.sf_id = record.sf_id
        factor.sf_name = record.factor_name
        factor.sf_explain = record.factor_explain
        factor.sf_source = record.factor_source
        factor.sf_kind = record.factor_kind
        factor.sf_formula = record.factor_formula
        factor.sf_start_date = record.start_date

        session.merge(factor)

    session.commit()
    session.close()








