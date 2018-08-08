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
from pyspark import SparkContext


logger = logging.getLogger(__name__)

def exposure_update(sf):
    sf.cal_factor_exposure()
    asset_stock_factor.update_exposure(sf)

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
    print('load all stock done')
    StockAsset.all_stock_quote()
    print('load all quote done')
    StockAsset.all_stock_fdmt()
    print('load all fdmt done')

    sfs = [
        SizeStockFactor(factor_id = "SF.000001"),
        VolStockFactor(factor_id = "SF.000002"),
        MomStockFactor(factor_id = "SF.000003"),
        TurnoverStockFactor(factor_id = "SF.000004"),
        EarningStockFactor(factor_id = "SF.000005"),
        ValueStockFactor(factor_id = "SF.000006"),
        FqStockFactor(factor_id = "SF.000007"),
        LeverageStockFactor(factor_id = "SF.000008"),
        GrowthStockFactor(factor_id = "SF.000009"),
        # FarmingStockFactor(factor_id = 'SF.100001'),
        # MiningStockFactor(factor_id = 'SF.100002'),
        # ChemicalStockFactor(factor_id = 'SF.100003'),
        # FerrousStockFactor(factor_id = 'SF.100004'),
        # NonFerrousStockFactor(factor_id = 'SF.100005'),
        # ElectronicStockFactor(factor_id = 'SF.100006'),
        # CTEquipStockFactor(factor_id = 'SF.100007'),
        # HouseholdElecStockFactor(factor_id = 'SF.100008'),
        FoodBeverageStockFactor(factor_id = 'SF.100009'),
        # TextileStockFactor(factor_id = 'SF.100010'),
        # LightIndustryStockFactor(factor_id = 'SF.100011'),
        MedicalStockFactor(factor_id = 'SF.100012'),
        # PublicStockFactor(factor_id = 'SF.100013'),
        # ComTransStockFactor(factor_id = 'SF.100014'),
        # RealEstateStockFactor(factor_id = 'SF.100015'),
        # TradingStockFactor(factor_id = 'SF.100016'),
        # TourismStockFactor(factor_id = 'SF.100017'),
        BankStockFactor(factor_id = 'SF.100018'),
        FinancialStockFactor(factor_id = 'SF.100019'),
        # CompositeStockFactor(factor_id = 'SF.100020'),
        # ConstructionStockFactor(factor_id = 'SF.100021'),
        # ArchitecturalStockFactor(factor_id = 'SF.100022'),
        # ElecEquipStockFactor(factor_id = 'SF.100023'),
        # MachineryStockFactor(factor_id = 'SF.100024'),
        # MilitaryStockFactor(factor_id = 'SF.100025'),
        # ComputerStockFactor(factor_id = 'SF.100026'),
        # MedicalStockFactor(factor_id = 'SF.100027'),
        # CommunicationStockFactor(factor_id = 'SF.100028'),
    ]

    # sfs = [
    #     FqStockFactor(factor_id = "SF.000007"),
    # ]

    # for _sf in sfs:
    #     _sf.cal_factor_exposure()
    #     print(_sf.factor_id, 'cal factor exposure done')
    #     asset_stock_factor.update_exposure(_sf)
    #     print(_sf.factor_id, 'update done')


    pool = Pool(len(sfs))
    pool.map(exposure_update, sfs)
    pool.close()
    pool.join()


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

    # StockAsset.all_stock_nav()
    # print('load all stock done')
    # StockAsset.all_stock_quote()
    # print('load all quote done')
    # StockAsset.all_stock_fdmt()
    # print('load all fdmt done')

    #sfs = ['SF.0000%02d'%i for i in range(1, 10)] + ['SF.1000%02d'%i for i in range(1, 29)]
    sfs = ['SF.0000%02d'%i for i in range(1, 10)]
    sf = StockFactor()
    df_ret, df_sret = sf.cal_factor_return(sfs)

    asset_stock_factor.update_stock_factor_return(df_ret)
    asset_stock_factor.update_stock_factor_specific_return(df_sret)













