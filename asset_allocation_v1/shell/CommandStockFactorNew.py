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
import time
from ipdb import set_trace

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund
from db.asset_stock_factor import *
from db.asset_stock import *
from stock_factor import *
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool


logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def sf(ctx):
    '''multi factor
    '''
    pass


@sf.command()
@click.pass_context
def factor_valid_update(ctx):
    '''valid stock factor update
    '''

    StockFactor.valid_stock_table()



@sf.command()
@click.pass_context
def factor_exposure_update(ctx):
    '''insert factor info
    '''

    StockAsset.all_stock_nav()
    print 'load all stock done'
    StockAsset.all_stock_quote()
    print 'load all quote done'
    StockAsset.all_stock_fdmt()
    print 'load all fdmt done'

    sfs = [
        #SizeStockFactor(factor_id = "SF.000001"),
        #VolStockFactor(factor_id = "SF.000002"),
        #MomStockFactor(factor_id = "SF.000003"),
        #TurnoverStockFactor(factor_id = "SF.000004"),
        #EarningStockFactor(factor_id = "SF.000005"),
        #ValueStockFactor(factor_id = "SF.000006"),
        #FqStockFactor(factor_id = "SF.000007"),
        #LeverageStockFactor(factor_id = "SF.000008"),
        SizeNlStockFactor(factor_id = "SF.000009"),
    ]

    for _sf in sfs:
        _sf.cal_factor_exposure()
        print _sf.factor_id, 'cal factor exposure done'
        asset_stock_factor.update_exposure(_sf)
        print _sf.factor_id, 'update done'

    #pool = Pool(len(sfs))
    #pool.map(update_exposure, zip(exposures, sf_ids))
    #pool.close()
    #pool.join()



@sf.command()
@click.pass_context
def factor_return_update(ctx):
    '''insert factor info
    '''

    # sfs = [
    #     SizeStockFactor(factor_id = "SF.000001"),
    #     VolStockFactor(factor_id = "SF.000002"),
    #     MomStockFactor(factor_id = "SF.000003"),
    #     TurnoverStockFactor(factor_id = "SF.000004"),
    #     EarningStockFactor(factor_id = "SF.000005"),
    #     ValueStockFactor(factor_id = "SF.000006"),
    #     FqStockFactor(factor_id = "SF.000007"),
    #     LeverageStockFactor(factor_id = "SF.000008"),
    #     SizeNlStockFactor(factor_id = "SF.000009"),
    # ]

    #     sfr = sf.ret
    #     sfr.index.name = 'trade_date'
    #     df_new = sfr.reset_index()
    #     df_new['sf_id'] = sf.factor_id
    #     df_new = df_new.set_index(['sf_id', 'trade_date'])

    #     df_old = load_stock_factor_return(sf_id = sf.factor_id)
    #     db = database.connection('asset')
    #     t = Table('stock_factor_return', MetaData(bind=db), autoload = True)
    #     database.batch(db, t, df_new, df_old)

    # pool = Pool(len(sfs))
    # pool.amap(save_return, sfs)
    # pool.close()
    # pool.join()

    sf = StockFactor()
    df_ret, df_sret = sf.cal_factor_return(['SF.000001', 'SF.000002', 'SF.000003', 'SF.000004','SF.000005', 'SF.000006','SF.000007', 'SF.000008'])

    asset_stock_factor.update_stock_factor_return(df_ret)
    asset_stock_factor.update_stock_factor_specific_return(df_sret)
