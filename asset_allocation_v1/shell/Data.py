#coding=utf8


import os
import string
import pandas as pd
from datetime import datetime
from numpy import *
import numpy as np
import Const
from Const import datapath


#start_date = '2014-03-07'
#end_date = '2015-02-09'
#index_col = '000300.SH'



def funds():
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/fund_value.csv')), index_col = 'date', parse_dates = ['date'] )
    return df


def bonds():
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/bond_value.csv')), index_col = 'date', parse_dates = ['date'] )
    return df


def moneys():
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/money_value.csv')), index_col = 'date', parse_dates = ['date'] )
    return df


def others():
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/other_value.csv')), index_col = 'date', parse_dates = ['date'] )
    return df


def stockindex():
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/fund_value.csv')), index_col = 'date', parse_dates = ['date'] )
    df = df[[const.hs300_code, const.largecap_code, const.smallcap_code, const.largecapgrowth_code, const.largecapvalue_code, const.smallcapvalue_code, const.smallcapgrowth_code, const.zz500_code]]
    return df


#基金和指数数据抽取和对齐
def fund_index_data(start_date, end_date, index_code):


    #取开始时间和结束时间的数据
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/fund_value.csv')), index_col = 'date', parse_dates = ['date'] )
    df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
    df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]


    #取基金成立时间指标
    indicator_df = pd.read_csv(os.path.normpath(datapath( '../csvdata/fund_establish_date.csv')), index_col = 'code', parse_dates = ['date'])
    establish_date_code = set()
    for code in indicator_df.index:
        date = indicator_df['establish_date'][code]
        if date <= datetime.strptime(start_date, '%Y-%m-%d'):
            establish_date_code.add(code)


    cols = df.columns
    fund_cols = []
    for col in cols:

        #有20%的净值是nan，则过滤掉该基金
        vs = df[col]
        n = 0
        for v in vs:
            if isnan(v):
                n = n + 1
        if n > 0.2 * len(vs):
            continue

        if col.find('OF') >= 0 and col in establish_date_code:
            fund_cols.append(col)



    fund_df = df[fund_cols]
    #fund_df_r = df_r[fund_cols]


    index_df = df[index_code]
    #index_df_r = df_r[index_col]

    #print fund_df_r
    #print index_df_r

    return fund_df, index_df



def fund_value(start_date, end_date):


    #取开始时间和结束时间的数据
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/fund_value.csv')), index_col = 'date', parse_dates = ['date'])
    df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
    df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]



    #取基金成立时间指标
    indicator_df = pd.read_csv(os.path.normpath(datapath( '../csvdata/fund_establish_date.csv')), index_col = 'code', parse_dates = ['date'])
    indicator_df = indicator_df.dropna()
    establish_date_code = set()
    for code in indicator_df.index:
        date = indicator_df['establish_date'][code]
        if datetime.strptime(date, '%Y-%m-%d') <= datetime.strptime(start_date, '%Y-%m-%d'):
            establish_date_code.add(code)


    cols = df.columns
    fund_cols = []
    for col in cols:

        #有20%的净值是nan，则过滤掉该基金
        vs = df[col].values
        n = 0
        for v in vs:
            if isnan(v):
                n = n + 1
        if n > 0.2 * len(vs):
            continue

        if col.find('OF') >= 0 and col in establish_date_code:
            fund_cols.append(col)


    fund_df = df[fund_cols]
    #funddf['163001.OF'].to_csv(datapath('163001.csv'))
    return fund_df


def bond_value(start_date, end_date):

    #取开始时间和结束时间的数据
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/bond_value.csv')), index_col = 0, parse_dates = ['date'])
    df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
    df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]

    #print df
    #取基金成立时间指标
    indicator_df = pd.read_csv(os.path.normpath(datapath( '../csvdata/bond_establish_date.csv')), index_col = 'code', parse_dates = ['date'])
    indicator_df = indicator_df.dropna()
    establish_date_code = set()
    for code in indicator_df.index:
        date = indicator_df['establish_date'][code]
        if datetime.strptime(date,'%Y-%m-%d') <= datetime.strptime(start_date, '%Y-%m-%d'):
            establish_date_code.add(code)

    cols = df.columns
    fund_cols = []
    for col in cols:

        #有20%的净值是nan，则过滤掉该基金
        vs = df[col].values
        n = 0
        for v in vs:
            if isnan(v):
                n = n + 1
        if n > 0.2 * len(vs):
            continue

        if col.find('OF') >= 0 and col in establish_date_code:
            fund_cols.append(col)

    fund_cols = list(set(fund_cols))

    fund_df = df[fund_cols]

    return fund_df


def money_value(start_date, end_date):


    #取开始时间和结束时间的数据
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/money_value.csv')), index_col = 0, parse_dates = ['date'])
    df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
    df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]


    #print df
    #取基金成立时间指标
    indicator_df = pd.read_csv(os.path.normpath(datapath( '../csvdata/money_establish_date.csv')), index_col = 'code', parse_dates = ['date'])
    indicator_df = indicator_df.dropna()
    establish_date_code = set()
    for code in indicator_df.index:
        date = indicator_df['establish_date'][code]
        if datetime.strptime(date,'%Y-%m-%d') <= datetime.strptime(start_date, '%Y-%m-%d'):
            establish_date_code.add(code)


    cols = df.columns
    fund_cols = []
    for col in cols:

        #有20%的净值是nan，则过滤掉该基金
        vs = df[col].values
        n = 0
        for v in vs:
            if isnan(v):
                n = n + 1
        if n > 0.2 * len(vs):
            continue

        if col.find('OF') >= 0 and col in establish_date_code:
            fund_cols.append(col)

    fund_cols = list(set(fund_cols))

    fund_df = df[fund_cols]

    return fund_df

def index_value(start_date, end_date, index_code):

    #取开始时间和结束时间的数据
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/fund_value.csv')), index_col = 'date', parse_dates = ['date'] )
    df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
    df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]

    index_df = df[index_code]

    return index_df

def bond_index_value(start_date, end_date, index_code):

    #取开始时间和结束时间的数据
    df = pd.read_csv(os.path.normpath(datapath( '../csvdata/bond_value.csv')), index_col = 'date', parse_dates = ['date'] )
    df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
    df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]

    #print index_code
    #print df
    index_df = df[index_code]

    return index_df



def establish_data():

    indicator_df = pd.read_csv(os.path.normpath(datapath( '../csvdata/fund_establish_date.csv')), index_col = 'code', parse_dates = ['date'])
    indicator_df = indicator_df.dropna()
    return indicator_df

def bond_establish_data():

    indicator_df = pd.read_csv(os.path.normpath(datapath( '../csvdata/bond_establish_date.csv')), index_col = 'code', parse_dates = ['date'])
    return indicator_df


def scale_data():
    indicator_df = pd.read_csv(os.path.normpath(datapath( '../csvdata/fund_scale.csv')), index_col = 'code')
    return indicator_df



def stock_fund_code():

    funddf = pd.read_csv(os.path.normpath(datapath( '../csvdata/stock_fund_code.csv')), index_col = 'code')
    codes = []
    for code in funddf.index:
        codes.append(code)
    return codes



def fund_position(start_date, end_date):


    positiondf = pd.read_csv(os.path.normpath(datapath( '../csvdata/fund_position.csv')), index_col = 'date' , parse_dates = ['date'])
    positiondf = positiondf[ positiondf.index <= datetime.strptime(end_date,'%Y-%m-%d')]
    positiondf = positiondf[ positiondf.index >= datetime.strptime(start_date,'%Y-%m-%d')]

    codes = []

    for col in positiondf.columns:
        vs = positiondf[col].values
        has = True
        for v in vs:
            try:
                if isnan(v):
                    has = False
            except:
                has = False

        if has:
            codes.append(col)


    positiondf = positiondf[codes]
    return positiondf


if __name__ == '__main__':

    #fund_df, index_df = fund_index_data('2011-02-03', '2015-03-02', '000300.SH')
    #print fund_df, index_df
    #print np.mean(index_df.pct_change())
    #fund_scale =  scale_data()
    #print fund_scale.values
    #print stock_fund_code()
    #print fund_position('2011-01-02','2012-12-31')
    fund, df = fund_index_data('2009-10-10','2016-04-22', ['000300.SH','000905.SH'])
    df.to_csv(datapath('index.csv'))
    #print df['000300.SH','000905.SH']
    #buysell()

