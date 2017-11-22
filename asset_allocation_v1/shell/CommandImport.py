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

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from tabulate import tabulate
from db import *
from util.xdebug import dd

import traceback, code

logger = logging.getLogger(__name__)

@click.group(name='import')  
@click.pass_context
def import_command(ctx):
    ''' generate portfolios
    '''
    pass;
    

@import_command.command()
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.option('--id', 'optid', help=u'questionare to import')
@click.option('--question', '-q', type=click.Path(), help=u'excel file for question')
@click.pass_context
def fp_da_question(ctx, optlist, optid, question):
    '''import mapi.fp_da_question from excel
    '''

    columns = {
        u'问题ID':'globalid',
        u'问卷ID':'fp_nare_id',
        u'问题标题':'fp_qst_text',
        u'问题副标题':'fp_qst_subtext',
        u'所属类别':'fp_qst_category',
        u'问题类型(1:填空；2:选择)':'fp_qst_type',
        u'是否必填':'fp_qst_required',
        u'数字数据下限':'fp_lower_limit',
        u'数字数据上限':'fp_upper_limit',
        # u'数据提示下限':'fp_',
        # u'数据提示文本（下限）':'fp_',
        # u'数据提示上限':'fp_',
        # u'数据提示文本（上限）':'fp_',
        u'备注':'fp_qst_remark',
    }

    df = pd.read_excel(question)
    df = df.rename(columns=columns)
    df = df[columns.values()].copy()

    df.set_index([u'globalid'], inplace=True)

    dd("aa", df.head())
    # db = database.connection('asset')

    # sql = "SELECT is_investor_id, DATE_FORMAT(is_date, '%%Y-%%m') AS is_month,SUM(is_return) as is_return FROM `is_investor_holding` WHERE 1 "
    # if optstartid is not None:
    #     sql += " AND is_investor_id >= '%s' " % optstartid
    # if optendid is not None:
    #     sql += " AND is_investor_id <= '%s' " % optendid
    # sql += "GROUP BY is_investor_id, is_month"

    # df_result = pd.read_sql(sql, db,  index_col=['is_investor_id', 'is_month'])
    # df_result = df_result.unstack(1)
    # df_result.columns = df_result.columns.droplevel(0)

    # if output is not None:
    #     path = output
    # else:
    #     path = datapath('import-nav.csv')
        
    # df_result.to_csv(path)

    # print "import nav to file %s" % (path)

