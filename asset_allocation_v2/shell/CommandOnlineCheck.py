# coding=utf-8
import sys
sys.path.append('shell')
import numpy as np
import pandas as pd
from ipdb import set_trace
import time
from datetime import datetime,timedelta,date
from dateutil.parser import parse
from sqlalchemy import MetaData,Table,select,literal_column
from trade_date import ATradeDate
from db import database,asset_mz_markowitz_asset,base_ra_index_nav
from asset import Asset
import logging
import warnings
import click
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def mz_asset_nav_check(targetDate,markowitz_id='MZ.000070'):
    '''
    markowitz --id MZ.000007 检查：MZ.000070配置指数及因子 最新数据是否更新
    '''

    print('\033[1;35mmarkowitz配置更新################################## Check\033[0m')

    startDate = (targetDate + timedelta(weeks=-2)).strftime('%Y-%m-%d')
    endDate = date.today().strftime('%Y-%m-%d')
    assets_id = asset_mz_markowitz_asset.load([markowitz_id])['mz_asset_id'].tolist()
    assets = dict([(asset_id , Asset(asset_id).nav(begin_date=startDate,end_date=endDate)) for asset_id in assets_id])
    df_assets = pd.DataFrame(assets)
    if df_assets.loc[targetDate].isnull().any():
        print('markowitz配置资产或因子数据更新：\033[1;43mFail inspection!!!\033[0m')
        print(df_assets.loc[[targetDate]])
        return False
    else:
        print('markowitz配置资产或因子数据更新：\033[1;35mPass inspection\033[0m')
        return True


def macroview_check(targetDate, markowitz_id = 'MZ.000070', weeks_lookback=52):
    '''
    ra_bl_view表  检查：宏观观点 最新数据是否更新
    '''

    print('\033[1;35m宏观观点################################## Check\033[0m')

    gids = asset_mz_markowitz_asset.load([markowitz_id])['mz_asset_id'].tolist()
    # 获取ra_blview表格的宏观观点数据
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_bl_view',metadata,autoload=True)
    columns = [
        t1.c.globalid,
        t1.c.bl_date,
        t1.c.bl_index_id,
        t1.c.bl_view,
    ]
    s = select(columns)
    startDate = (targetDate + timedelta(weeks=-1 * weeks_lookback)).strftime('%Y-%m-%d')
    s = s.where(t1.c.bl_date >= startDate).where(t1.c.globalid == 'BL.000001')
    df = pd.read_sql(s,db,parse_dates=['bl_date'], index_col = ['bl_index_id', 'bl_date'])
    gids.append('ALayer')
    try:
        for asset_id in gids:
            v = df.loc[asset_id]
            v = v.sort_index(ascending=True)
            print(asset_id, v.index[-1], v.bl_view[-1])
        print('markowitz配置资产宏观观点都存在，请检查观点是否正确：\033[1;35mPass inspection\033[0m')
        return True
    except:
        print('markowitz配置资产宏观观点缺失：\033[1;43mFail inspection!!!\033[0m')
        return False


def cov_check(targetDate, markowitz_id = 'MZ.000070'):
    '''
    on_online_cov表，检查：资产或因子 的最新数据是否完整且离谱
    '''

    print('\033[1;35m协方差#################################### Check\033[0m')

    # 获取on_online_cov表格的数据
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('on_online_cov',metadata,autoload=True)
    columns = [
        t1.c.on_asseta_id,
        t1.c.on_assetb_id,
        t1.c.on_date,
        t1.c.on_cov,
    ]
    s = select(columns)
    s = s.where(t1.c.on_date == targetDate)
    df = pd.read_sql(s,db,parse_dates=['on_date'])

    gids = asset_mz_markowitz_asset.load([markowitz_id])['mz_asset_id'].tolist()
    if any(df) and df['on_cov'].isnull().any()==False:
        print('协方差数据更新：\033[1;35mPass inspection\033[0m')
        if len(df[df.on_asseta_id==df.on_assetb_id])**2 == len(df) and len(df) == len(gids) ** 2:
            print('协方差矩阵条数：\033[1;35mPass inspection\033[0m')
        else:
            print('协方差矩阵条数：\033[1;43mFail inspection!!!\033[0m')
        df_covariance = df[df.on_asseta_id==df.on_assetb_id]
        if (df_covariance['on_cov']==0.0).any():
            print('方差非零：\033[1;43mFail inspection!!!\033[0m')
            all_df = df_covariance[df_covariance['on_cov']==0].pivot(index='on_date',columns='on_asseta_id',values='on_cov')
            del all_df.index.name
            print(all_df)
            return False
        else:
            print('方差非零：\033[1;43mPass inspection!!!\033[0m')
        df_covariance['stock'] = df_covariance['on_asseta_id'].map(lambda x: int(x) if str(x)[:4] in ['1111'] else None)
        df_covariance['bond'] = df_covariance['on_asseta_id'].map(lambda x: int(x) if str(x)[:4] in ['1121'] else None)
        stock_cov = df_covariance[~(df_covariance['stock'].isnull())]['on_cov'].tolist()
        bond_cov = df_covariance[~(df_covariance['bond'].isnull())]['on_cov'].tolist()
        if max(stock_cov) > min(stock_cov)*4:
            print('国内股票方差检查：\033[1;43mFail inspection!!!\033[0m')
            stock_df = df_covariance[~(df_covariance['stock'].isnull())].pivot(index='on_date',columns='on_asseta_id',values='on_cov')
            del stock_df.index.name

            print(stock_df)
        else:
            print('国内股票方差检查：\033[1;35mPass inspection\033[0m')
        if max(bond_cov) > min(bond_cov)*4:
            print('国内债券方差正常：\033[1;43mFail inspection!!!\033[0m')
            bond_df = df_covariance[~(df_covariance['bond'].isnull())].pivot(index='on_date',columns='on_asseta_id',values='on_cov')
            del bond_df.index.name
            print(bond_df)
        else:
            print('国内债券方差正常：\033[1;35mPass inspection\033[0m')
    else:
        print('协方差数据更新：\033[1;43mFail inspection!!!\033[0m')

    return True


def ra_pool_fund_check(targetDate,weeks_lookback=1000):
    '''
    # 检查基金池：基金不能在不同基金池出现 <ra_portfolio_asset,ra_pool_fund表里，PO.000007记录基金池>
    '''
    # 获取ra_pool_fund表格的数据
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_pool_fund',metadata,autoload=True)
    columns = [
        t1.c.ra_date,
        t1.c.ra_pool,
        t1.c.ra_fund_id,
    ]
    s = select(columns)
    startDate = (targetDate + timedelta(weeks=-1*weeks_lookback)).strftime('%Y-%m-%d')
    s = s.where(t1.c.ra_date >= startDate)
    s = s.where(t1.c.ra_date <= targetDate)
    df = pd.read_sql(s,db,parse_dates=['ra_date']).set_index('ra_date')
    df = df.loc[max(df.index)].reset_index()
    if (df.groupby('ra_fund_id')['ra_pool'].count()==1).all():
        print('基金池是否交叉：\033[1;35mPass inspection\033[0m 更新日期：{}'.format(max(df['ra_date'])))
    else:
        print('基金池是否交叉：\033[1;43mFail inspection!!!\033[0m 更新日期：{}'.format(max(df['ra_date'])))
    pass


def turnover_check(targetDate,weeks_lookback=13):
    '''
    检查on_online_markowitz 表里最近一期和上一期的每个风险等级的换手率
    '''
    # 获取on_online_markowitz表格的数据
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('on_online_markowitz',metadata,autoload=True)
    columns = [
        t1.c.on_date,
        t1.c.on_online_id,
        #t1.c.on_asset_risk,
        t1.c.on_asset_id,
        t1.c.on_ratio,
        t1.c.on_online_status,
    ]
    s = select(columns)
    startDate = (targetDate + timedelta(weeks=-1*weeks_lookback)).strftime('%Y-%m-%d')
    s = s.where(t1.c.on_date >= startDate)
    s = s.where(t1.c.on_date <= targetDate)
    df = pd.read_sql(s,db,parse_dates=['on_date']).set_index('on_date')
    print('on_online_markowitz更新日期：{1}，目标日期：{0}'.format(targetDate,max(df.index)))
    df_pre = df.loc[max(df.index)]
    df_online = df.drop(index=max(df.index))
    if (df_pre['on_online_status']==1).all() and (df_online['on_online_status']==5).all():
        print('换手率数据预上线：\033[1;35mPass inspection\033[0m')
        turnover_all = []
        for name,group in df.groupby(by=['on_online_id']):
            df_group = group.pivot(columns='on_asset_id',values='on_ratio')
            turnover = (df_group - df_group.shift(1)).abs().sum(axis=1)
            turnover_all.append((name,turnover))
        df_turnover = pd.DataFrame(dict(turnover_all)).loc[[max(df.index)]]
        del df_turnover.index.name
        print(df_turnover)
    else:
        print('换手率数据预上线：\033[1;43mFail inspection!!!\033[0m')

    #turnover_all = []
    #for name,group in df.groupby(by=['on_online_id']):
    #    df_group = group.pivot(columns='on_asset_id',values='on_ratio')
    #    turnover = (df_group - df_group.shift(1)).abs().sum(axis=1)
    #    turnover_all.append((name,turnover))
    #df_turnover = pd.DataFrame(dict(turnover_all)).loc[[max(df.index)]]
    #del df_turnover.index.name
    #print(df_turnover)

    return None


def obj_date(datestr=None):
    '''
    目标日期生成: 如，输入‘2019-04-12’
    '''
    if datestr is not None:
        startDate = datestr
    else:
        startDate = (date.today() + timedelta(weeks=-2)).strftime('%Y-%m-%d')
    targetDate = ATradeDate.week_trade_date(begin_date = startDate)[-1]
    return targetDate

@click.group(invoke_without_command=True)
@click.pass_context
def check(ctx):
    '''
        online check
    '''
    targetDate = obj_date()
    print('最近周的最后一个交易日 : ' + targetDate.strftime('%Y-%m-%d'))
    print()

    #检查markowitz配置的资产最后一个周交易日的数据是否存在
    if mz_asset_nav_check(targetDate):
        pass
    else:
        pass
    print()

    #检查宏观观点
    if macroview_check(targetDate):
        pass
    else:
        pass
    print()

    #检查协方差矩阵
    if cov_check(targetDate):
        pass
    else:
        #return
        pass
    print()


    #基金池，看是否有一只基金在两个基金池中出现
    if ra_pool_fund_check(targetDate):
        pass
    else:
        #return
        pass
    print()


    #查看换手率
    if turnover_check(targetDate):
        pass
    else:
        pass
    print()


if __name__ == '__main__':
    pass
