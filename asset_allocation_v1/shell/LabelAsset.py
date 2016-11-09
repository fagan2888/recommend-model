#coding=utf8


import numpy as np
import string
import os
import sys
import click
sys.path.append("windshell")
import Financial as FIN
import StockTag as ST
import Data
from numpy import *
import datetime
import FundFilter
import FundIndicator
import FundSelector
import Const
import pandas as pd
import time
import AllocationData
import DBData
import DFUtil

from Const import datapath
from dateutil.parser import parse


def label_asset_tag(label_index, lookback=52):
    '''perform fund tagging along label_index.
    '''
    # label_index = pd.DatetimeIndex(['2012-10-26', '2015-09-30', '2016-04-08', '2016-10-14'])
    label_index = pd.DatetimeIndex(['2016-10-14'])
    

    label_index = pd.DatetimeIndex(['2014-03-31','2014-06-30','2014-09-30','2014-12-31', '2015-03-31','2015-06-30','2015-09-30','2015-12-31','2016-03-31','2016-06-30', '2016-09-30'])

    #
    # 计算每个调仓点的最新配置
    #
    data_stock = {}
    data_bond = {}
    data_money = {}
    data_other = {}
    with click.progressbar(length=len(label_index), label='label asset') as bar:
        for day in label_index:
            data_stock[day] = label_asset_stock_per_day(day, lookback, Const.fund_num)
            data_bond[day] = label_asset_bond_per_day(day, lookback, Const.fund_num)
            data_money[day] = label_asset_money_per_day(day, lookback, Const.fund_num)
            data_other[day] = label_asset_other_per_day(day, lookback, Const.fund_num)
            bar.update(1)
        
    df_stock = pd.concat(data_stock, names=['date', 'category', 'code'])
    df_bond = pd.concat(data_bond, names=['date', 'category', 'code'])
    df_money = pd.concat(data_money, names=['date', 'category', 'code'])
    df_other = pd.concat(data_other, names=['date', 'category', 'code'])

    df_result = pd.concat([df_stock, df_bond, df_money, df_other])
    df_result = df_result.swaplevel(0, 1)
    df_result.sort_index(inplace=True)
    df_result.to_csv(datapath('fund_pool.csv'))

    #
    # 兼容老版本的基金池
    #
    fund_pool_convert_to_old(df_stock, datapath('stock_fund.csv'))
    fund_pool_convert_to_old(df_bond, datapath('bond_fund.csv'))
    fund_pool_convert_to_old(df_money, datapath('money_fund.csv'))
    fund_pool_convert_to_old(df_other, datapath('other_fund.csv'))

def fund_pool_convert_to_old(df, path):
    df.reset_index(2, inplace=True)
    df = df.swaplevel(0, 1)
    data = {}
    for category in df.index.levels[0]:
        sr_category = df.loc[category]['code']
        data[category] = sr_category.groupby(level=0).apply(lambda x: list(x))

    df_result = pd.DataFrame(data)
    df_result.to_csv(path)


def label_asset_nav(start_date, end_date):
    df_pool = pd.read_csv(datapath('fund_pool.csv'),  index_col=['category', 'date', 'code'], parse_dates=['date'])
    df_pool['ratio'] = 1.0

    data = {}
    with click.progressbar(length=len(df_pool.index.levels[0]), label='label nav') as bar:
        for category in df_pool.index.levels[0]:
            #
            # 生成大类的基金仓位配置矩阵
            #
            df_pool_category = df_pool.loc[category, ['ratio']]
            df_position = df_pool_category.groupby(level=0).apply(lambda x: x / len(x))
            df_position = df_position.unstack(fill_value=0.0)
            df_position.columns = df_position.columns.droplevel(0)
            #
            # 加载各个基金的日净值数据
            #
            if category in ['GLNC', 'HSCI.HI', 'SP500.SPI']:
                df_nav_fund = DBData.db_index_value_daily(start_date, end_date, df_position.columns)
            else:
                df_nav_fund = DBData.db_fund_value_daily(start_date, end_date, codes=df_position.columns)
            df_inc_fund = df_nav_fund.pct_change().fillna(0.0)
            #
            # 计算组合净值增长率
            #
            df_nav_portfolio = DFUtil.portfolio_nav(df_inc_fund, df_position, result_col='portfolio')
            df_nav_portfolio.to_csv(datapath('category_nav_' + category + '.csv'))

            data[category] = df_nav_portfolio['portfolio']
            bar.update(1)

    df_result = pd.DataFrame(data)
    df_result.to_csv(datapath('labelasset.csv'))

    return df_result
        

def label_asset_stock_per_day(day, lookback, limit = 5):
    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)

    start_date = index.min().strftime("%Y-%m-%d")
    end_date = day.strftime("%Y-%m-%d")

    # 加载数据
    df_nav_stock = DBData.stock_fund_value(start_date, end_date)
    df_nav_index = DBData.index_value(start_date, end_date)
    
    df_nav_stock.to_csv(datapath('stock_' + day.strftime('%Y-%m-%d') + '.csv'))
    
    #
    # 根据时间轴进行重采样
    #
    df_nav_stock = df_nav_stock.reindex(index, method='pad')
    df_nav_index = df_nav_index.reindex(index, method='pad')

    # #
    # # 计算涨跌幅
    # #
    # df_inc_stock = df_nav_stock.pct_change().fillna(0.0)
    # df_inc_index = df_nav_index.pct_change().fillna(0.0)

    #
    # 基于测度筛选基金
    #
    df_indicator = FundFilter.stock_fund_filter_new(
        day, df_nav_stock, df_nav_index[Const.hs300_code])

    #print df_nav_stock
    codes = set(df_nav_stock.columns.values)
    print 'end_date', end_date
    print 'fund' , len(codes)
    #print codes
    #
    # 打标签确定所有备选基金
    #
    #code_df = pd.read_csv('./data/codes.csv')
    #code    = []
    #for c in code_df['0'].values:
    #    code.append('%06d' % c)
    #codes = code_df.values
    #print 'bbbbbbb', codes
    #df_nav_indicator = df_nav_stock[df_indicator.index]
    #print df_nav_indicator
    #code = set(df_nav_stock.columns.values)
    #code = list(code)
    #print code
    #print end_date
    #print '-----------------', len(df_nav_stock.columns)

    fund_size_df = pd.read_csv('./data/fund_size.csv', index_col = 'date')
    fund_invshare_df = pd.read_csv('./data/fund_inv_ratio.csv', index_col = 'date')
    lines = open('data/totalyear').readlines()
    totalyear = {}
    for line in lines:
        vec = line.strip().split()
        totalyear[vec[0]] = (int)(vec[1])
    date_size_df = fund_size_df.loc[end_date]
    date_invshare_df = fund_invshare_df.loc[end_date]
    date_size_df = date_size_df.dropna()
    date_invshare_df = date_invshare_df.dropna()

    sizes = date_size_df.values
    sizes.sort()
    #u     = np.mean(sizes)
    #sigma = np.std(sizes)
    #print u, sigma, u - sigma
    #print sizes[-1 * (int)(0.2 * len(sizes))], sizes[-1]
    size_codes = []
    for code in date_size_df.index:
        size = date_size_df.loc[code]
        #print code, size
        if size >= sizes[-1 * (int)(0.3 * len(sizes))] and size <= sizes[-1 * (int)(0.1 * len(sizes))]:
            #print code, size
            size_codes.append(code)
    codes = codes & set(size_codes)
    print 'size done', len(codes)

    invshare_codes = []
    for code in date_invshare_df.index:
        invshare_ratio = date_invshare_df.loc[code]
        if invshare_ratio > 40:
            invshare_codes.append(code)

    codes = codes & set(invshare_codes)
    print 'invshare done' , len(codes)


    '''
    year_codes = []
    remain_codes = []
    vec = end_date.split('-')
    year = (int)(vec[0])
    for code, y in totalyear.items():
        if year == 2015:
            if y >= 7:
                year_codes.append(code)
            else:
                remain_codes.append(code)
        elif year == 2016:
            if y >= 6:
                year_codes.append(code)
            else:
                remain_codes.append(code)
    '''

    #codes = set(year_codes) & codes
    #print end_date, codes
    #print 'year done' , len(codes)

    #print codes

    #cs = []
    #for code in remain_codes:
    #    if code in set(codes):
    #        cs.append(code)
    #print end_date, cs
    #codes = set(cs) & codes
    #print end_date, codes
    #print end_date, len(codes)

    #codes = list(codes)
    #columns = set(df_nav_stock.columns.values)
    #codes = codes & columns
    codes = list(codes)
    #print len(codes)
    #print codes
    #print type(end_date)
    #print len(size_codes)
    #print len(date_size_df)
    #print len(date_invshare_df)
    #print end_date
    #print fund_invshare_df
    #print fund_size_df
    #df_nav_indicator = df_nav_stock[df_indicator.index]
    df_nav_indicator = df_nav_stock
    df_label = ST.tag_stock_fund_new(day, df_nav_indicator, df_nav_index)
    #codes = set(codes) & set(df_label.index.values) & set(df_indicator.index.values)
    #codes = list(codes)
    #print len(codes)
    #df_label = df_label.loc[codes]
    #df_indicator = df_indicator.loc[codes]

    #
    # 选择基金
    #
    df_stock_fund = FundSelector.select_stock_new(day, df_label, df_indicator, limit)

    return df_stock_fund

def label_asset_bond_per_day(day, lookback, limit = 5):
    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)

    start_date = index.min().strftime("%Y-%m-%d")
    end_date = day.strftime("%Y-%m-%d")

    # 加载数据
    df_nav_bond = DBData.bond_fund_value(start_date, end_date)
    df_nav_index = DBData.index_value(start_date, end_date)

    df_nav_bond.to_csv(datapath('bond_' + day.strftime('%Y-%m-%d') + '.csv'))
    
    #
    # 根据时间轴进行重采样
    #
    df_nav_bond = df_nav_bond.reindex(index, method='pad')
    df_nav_index = df_nav_index.reindex(index, method='pad')

    # #
    # # 计算涨跌幅
    # #
    # df_inc_bond = df_nav_bond.pct_change().fillna(0.0)
    # df_inc_index = df_nav_index.pct_change().fillna(0.0)

    #
    # 基于测度筛选基金
    #
    df_indicator = FundFilter.bond_fund_filter_new(
        day, df_nav_bond, df_nav_index[Const.csibondindex_code])

    #
    # 打标签确定所有备选基金
    #
    df_nav_indicator = df_nav_bond[df_indicator.index]
    df_label = ST.tag_bond_fund_new(day, df_nav_indicator, df_nav_index)

    #
    # 选择基金
    #
    # print day, df_label, df_indicator
    df_bond_fund = FundSelector.select_bond_new(day, df_label, df_indicator, limit)

    return df_bond_fund

def label_asset_money_per_day(day, lookback, limit = 1):
    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)

    start_date = index.min().strftime("%Y-%m-%d")
    end_date = day.strftime("%Y-%m-%d")

    # 加载数据
    df_nav_money = DBData.money_fund_value(start_date, end_date)

    df_nav_money.to_csv(datapath('money_' + day.strftime('%Y-%m-%d') + '.csv'))
    
    #
    # 根据时间轴进行重采样
    #
    df_nav_money = df_nav_money.reindex(index, method='pad')

    # #
    # # 计算涨跌幅
    # #
    # df_inc_money = df_nav_money.pct_change().fillna(0.0)
    # df_inc_index = df_nav_index.pct_change().fillna(0.0)

    #
    # 基于测度筛选基金
    #
    df_indicator = FundFilter.money_fund_filter_new(day, df_nav_money)

    #
    # 选择基金
    #
    df_money_fund = FundSelector.select_money_new(day, df_indicator, limit)

    return df_money_fund

def label_asset_other_per_day(day, lookback, limit):
    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)

    start_date = index.min().strftime("%Y-%m-%d")
    end_date = day.strftime("%Y-%m-%d")

    # 加载数据
    df_nav_other = DBData.other_fund_value(start_date, end_date)
    df_nav_other = df_nav_other[['SP500.SPI','GLNC','HSCI.HI']]
    df_nav_other.to_csv(datapath('other_' + day.strftime('%Y-%m-%d') + '.csv'))
    
    #
    # 根据时间轴进行重采样
    #
    df_nav_other = df_nav_other.reindex(index, method='pad')

    # #
    # # 计算涨跌幅
    # #
    # df_inc_other = df_nav_other.pct_change().fillna(0.0)

    #
    # 基于测度筛选基金
    #
    df_indicator = FundFilter.other_fund_filter_new(day, df_nav_other)

    #
    # 选择基金
    #
    df_other_fund = FundSelector.select_other_new(day, df_indicator)

    return df_other_fund


def mean_r(d, funddfr, codes):
    r   = 0.0
    num = 1.0 * (len(codes))
    for code in codes:
        r = r + funddfr.loc[d, code] / num

    return r


def stockLabelAsset(allocationdata, dates, his_week, interval):

    indexdf   = DBData.index_value(dates[0], dates[-1])

    tag = {}

    result_dates = []
    columns      = []
    result_datas = []
    select_datas = []

    fund_dates = []
    fund_datas = []

    allcodes    = []
    filtercodes = []
    poolcodes   = []
    selectcodes = []

    #print datapath("aa.csv")

    #print range(his_week, len(dates))
    for i in range(his_week, len(dates)):


        if (i - his_week) % interval == 0:
            print "aa", i-his_week, dates[i]

            start_date                    = dates[i - his_week].strftime('%Y-%m-%d')
            end_date                      = dates[i - 1].strftime('%Y-%m-%d')

            allocation_start_date         = dates[i - interval].strftime('%Y-%m-%d')

            #label_start_date             = dates[i - interval].strftime('%Y-%m-%d')

            last_end_date = dates[-1]
            if i + interval < len(dates):
                last_end_date                 = dates[i + interval].strftime('%Y-%m-%d')

            stock_df = DBData.stock_fund_value(start_date, last_end_date)
            label_stock_df = DFUtil.get_date_df(stock_df, start_date, end_date)
            this_index_df  = DFUtil.get_date_df(indexdf, start_date, end_date)
            funddfr = stock_df.pct_change().fillna(0.0)
            #df = (allocationdata.stock_df, allocation_start_date, end_date)
            #alldf        =

            #print
            #print time.time()
            
            label_stock_df.to_csv(datapath('stock_' + dates[i].strftime('%Y-%m-%d') + '.csv'))

            codes, indicator     = FundFilter.stockfundfilter(allocationdata, label_stock_df, indexdf[Const.hs300_code])
            #print time.time()
            #print '------------------------------------------------------------------------------------------'
            #print label_stock_df.index
            fund_pool, fund_tags = ST.tagstockfund(allocationdata, label_stock_df[codes], this_index_df)

            #print time.time()
            #print
            allocationdf   = DFUtil.get_date_df(label_stock_df[fund_pool], allocation_start_date, end_date)
            #fund_code, tag = FundSelector.select_stock(allocationdf, fund_tags)
            fund_code, tag = FundSelector.select_stock(label_stock_df, fund_tags, this_index_df[Const.hs300_code])


            allcodes    = label_stock_df.columns
            filtercodes = codes
            poolcodes   = fund_pool
            selectcodes = fund_code


            #print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']
            #print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']


            f = open(datapath('stock_pool_codes_' + end_date +'.txt'),'w')
            for code in poolcodes:
                f.write(str(code) + "\n")
            f.close()


            fund_dates.append(end_date)
            fund_datas.append([tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']])


        d = dates[i]
        result_dates.append(d)
        result_datas.append([ mean_r(d, funddfr, tag['largecap']), mean_r(d, funddfr , tag['smallcap']), mean_r( d, funddfr, tag['rise']), mean_r( d, funddfr, tag['oscillation']) , mean_r(d, funddfr, tag['decline']), mean_r(d, funddfr, tag['growth']), mean_r( d, funddfr, tag['value'])])
        print d.strftime('%Y-%m-%d'), mean_r(d, funddfr, tag['largecap']), mean_r(d, funddfr , tag['smallcap']), mean_r( d, funddfr, tag['rise']), mean_r( d, funddfr, tag['oscillation']) , mean_r(d, funddfr, tag['decline']), mean_r(d, funddfr, tag['growth']), mean_r( d, funddfr, tag['value'])


        allcode_r = 0
        for code in allcodes:
            allcode_r = allcode_r + 1.0 / len(allcodes) * funddfr.loc[d, code]


        filtercode_r = 0
        for code in filtercodes:
            filtercode_r = filtercode_r + 1.0 / len(filtercodes) * funddfr.loc[d, code]


        poolcode_r = 0
        for code in poolcodes:
            poolcode_r = poolcode_r + 1.0 / len(poolcodes) * funddfr.loc[d, code]

        selectcode_r = 0
        for code in selectcodes:
            selectcode_r = selectcode_r + 1.0 / len(selectcodes) * funddfr.loc[d, code]

        select_datas.append([allcode_r, filtercode_r, poolcode_r, selectcode_r])


    result_df = pd.DataFrame(result_datas, index = result_dates, columns=['largecap', 'smallcap', 'rise', 'oscillation', 'decline', 'growth', 'value'])
    result_df.to_csv(datapath('stocklabelasset.csv'))

    select_df = pd.DataFrame(select_datas, index = result_dates, columns=['allcodes','filtercodes','poolcode','selectcode'])
    select_df.to_csv(datapath('stockselectasset.csv'))

    fund_df = pd.DataFrame(fund_datas , index = fund_dates, columns=['largecap', 'smallcap', 'rise', 'oscillation', 'decline', 'growth', 'value'])
    fund_df.index.name = 'date'
    fund_df.to_csv(datapath('stock_fund.csv'))


    allocationdata.stock_fund_df = fund_df

    print 'stock label asset done'

    return result_df


def bondLabelAsset(allocationdata, dates, his_week, interval):


    #df  = Data.bonds()
    #dfr = df.pct_change().fillna(0.0)

    #funddfr = funddf.pct_change().fillna(0.0)
    #indexdfr = indexdf.pct_change().fillna(0.0)

    indexdf   = DBData.index_value(dates[0], dates[-1])

    pre_ratebond        = [u'217003']
    pre_creditbond      = [u'217003']
    pre_convertiblebond = [u'217003']

    tag = {}
    result_dates = []
    columns = []
    result_datas = []


    fund_dates = []
    fund_datas = []

    select_datas = []

    allcodes    = []
    filtercodes = []
    poolcodes   = []
    selectcodes = []


    for i in range(his_week, len(dates)):

        if (i - his_week) % interval == 0:

            start_date = dates[i - his_week].strftime('%Y-%m-%d')
            end_date = dates[i - 1].strftime('%Y-%m-%d')
            allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')

            last_end_date = dates[-1]
            if i + interval < len(dates):
                 last_end_date                 = dates[i + interval].strftime('%Y-%m-%d')

            last_end_date = '2015-04-03'

            bond_df = DBData.bond_fund_value(start_date, last_end_date)
            label_bond_df = DFUtil.get_date_df(bond_df, start_date, end_date)
            this_index_df = DFUtil.get_date_df(indexdf, start_date, end_date)
            funddfr = bond_df.pct_change().fillna(0.0)


            codes, indicator     = FundFilter.bondfundfilter(allocationdata, label_bond_df, indexdf[Const.csibondindex_code])
            fund_pool, fund_tags = ST.tagbondfund(allocationdata, label_bond_df[codes] ,this_index_df)


            allocationdf   = DFUtil.get_date_df(label_bond_df[fund_pool], allocation_start_date, end_date)
            #fund_code, tag = FundSelector.select_bond(allocationdf, fund_tags)
            fund_code, tag = FundSelector.select_bond(label_bond_df, fund_tags, this_index_df[Const.csibondindex_code])
            print tag

            if len(codes) == 1:
                fund_pool = codes
                fund_code = codes
                tag['ratebond']        = codes
                tag['creditbond']      = codes
                tag['convertiblebond'] = codes

            allcodes    = label_bond_df.columns
            filtercodes = codes
            poolcodes   = fund_pool
            selectcodes = fund_code

            if not tag.has_key('ratebond') or len(tag['ratebond']) == 0:
                tag['ratebond'] = pre_ratebond
            else:
                pre_ratebond    = tag['ratebond']
            if not tag.has_key('creditbond') or len(tag['creditbond']) == 0:
                tag['creditbond'] = pre_creditbond
            else:
                pre_creditbond  = tag['creditbond']
            if not tag.has_key('convertiblebond') or len(tag['convertiblebond']) == 0:
                tag['convertiblebond'] = pre_convertiblebond
            else:
                pre_convertiblebond = tag['convertiblebond']


            #print tag['ratebond'], tag['creditbond'], tag['convertiblebond']
            # print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']


            fund_dates.append(end_date)
            fund_datas.append([tag['ratebond'] , tag['creditbond'], tag['convertiblebond']])


            #f = open(datapath('bond_pool_codes_' + end_date + '.txt','w'))
            #for code in poolcodes:
            #    f.write(str(code) + "\n")
            #f.close()


        d = dates[i]
        result_dates.append(d)
        result_datas.append(
            [mean_r(d, funddfr, tag['ratebond']), mean_r(d, funddfr, tag['creditbond']), mean_r(d, funddfr, tag['convertiblebond'])])

        print d.strftime('%Y-%m-%d'), mean_r(d, funddfr, tag['ratebond']), mean_r(d, funddfr, tag['creditbond']), mean_r(d, funddfr, tag['convertiblebond'])


        allcode_r = 0
        for code in allcodes:
            allcode_r = allcode_r + 1.0 / len(allcodes) * funddfr.loc[d, code]

        filtercode_r = 0
        for code in filtercodes:
            filtercode_r = filtercode_r + 1.0 / len(filtercodes) * funddfr.loc[d, code]


        poolcode_r = 0
        for code in poolcodes:
            poolcode_r = poolcode_r + 1.0 / len(poolcodes) * funddfr.loc[d, code]


        selectcode_r = 0
        for code in selectcodes:
            selectcode_r = selectcode_r + 1.0 / len(selectcodes) * funddfr.loc[d, code]


        select_datas.append([allcode_r, filtercode_r, poolcode_r, selectcode_r])


    result_df = pd.DataFrame(result_datas, index=result_dates,
                             columns=['ratebond', 'creditbond', 'convertiblebond'])
    result_df.to_csv(datapath('bondlabelasset.csv'))

    select_df = pd.DataFrame(select_datas, index = result_dates, columns=['allcodes','filtercodes','poolcode','selectcode'])
    select_df.to_csv(datapath('bondselectasset.csv'))


    fund_df = pd.DataFrame(fund_datas , index = fund_dates, columns=['ratebond', 'creditbond','convertiblebond'])
    fund_df.index.name = 'date'
    #tmp_d = fund_df.index[-1]
    #fund_df.loc[tmp_d, 'ratebond'] = '200113'
    fund_df.to_csv(datapath('bond_fund.csv'))


    allocationdata.bond_fund_df = fund_df


    print 'bond label asset done'

    return result_df


def moneyLabelAsset(allocationdata, dates, his_week, interval):

    #funddfr = funddf.pct_change().fillna(0.0)

    tag = {}
    result_dates = []
    columns = []
    result_datas = []
    fund_dates   = []
    fund_datas   = []


    for i in range(his_week, len(dates)):


        if (i - his_week) % interval == 0:


            #start_date = dates[i - 52].strftime('%Y-%m-%d')
            #end_date = dates[i].strftime('%Y-%m-%d')
            #allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')

            #allocation_funddf = Data.money_value(allocation_start_date, end_date)
            #fund_codes, tag   = FundSelector.select_money(allocation_funddf)


            start_date = dates[i - his_week].strftime('%Y-%m-%d')
            end_date = dates[i - 1].strftime('%Y-%m-%d')
            allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')


            last_end_date = dates[-1]
            if i + interval < len(dates):
                 last_end_date                 = dates[i + interval].strftime('%Y-%m-%d')
                 
            last_end_date = '2015-04-03'


            money_df = DBData.money_fund_value(start_date, last_end_date)
            label_money_df = DFUtil.get_date_df(money_df, start_date, end_date)
            #this_index_df = DFUtil.get_date_df(indexdf, start_date, end_date)
            funddfr = money_df.pct_change().fillna(0.0)

            fund_codes, tag   = FundSelector.select_money(label_money_df)
            print tag

            #fund_sharpe       = FundIndicator.fund_sharp_annual(label_money_df)


            fund_dates.append(end_date)
            fund_datas.append(fund_codes[0])


            #print fund_sharpe[0][0]


        #print tag
        # print tag
        # print fund_codes

        d = dates[i]
        result_dates.append(d)
        result_datas.append(
            [funddfr.loc[d, tag['money']]])

        print d.strftime('%Y-%m-%d'), ',', funddfr.loc[d, tag['money']]


    result_df = pd.DataFrame(result_datas, index=result_dates,columns=['money'])
    result_df.to_csv(datapath('moneylabelasset.csv'))


    fund_df = pd.DataFrame(fund_datas, index=fund_dates, columns=['money'])
    fund_df.index.name = 'date'

    fund_df.to_csv(datapath('money_fund.csv'))
    allocationdata.money_fund_sharpe_df = fund_df

    print 'money label asset done'


    return result_df



def otherLabelAsset(allocationdata, dates, his_week, interval):


    start_date = dates[0]
    end_date   = dates[-1]
    funddf     = DBData.other_fund_value(start_date, end_date)
    funddfr    = funddf.pct_change().fillna(0.0)


    result_dates = []
    columns = []
    result_datas = []

    fund_dates   = []
    fund_datas   = []

    for i in range(his_week, len(dates)):

        d = dates[i]

        result_dates.append(d)
        result_datas.append(
            [funddfr.loc[d, 'SP500.SPI'], funddfr.loc[d, 'GLNC'], funddfr.loc[d, 'HSCI.HI']] )

        print d.strftime('%Y-%m-%d'), ',', funddfr.loc[d, 'SP500.SPI'], ',', funddfr.loc[d, 'GLNC'], ',' , funddfr.loc[d, 'HSCI.HI']

        if (i - his_week) % interval == 0:

            start_date = dates[i - his_week].strftime('%Y-%m-%d')
            end_date = dates[i - 1].strftime('%Y-%m-%d')

            label_other_df = DFUtil.get_date_df(funddf, start_date, end_date)

            fund_sharpe       = FundIndicator.fund_sharp_annual(label_other_df)

            sharpe_dict = {}
            for record in fund_sharpe:
                sharpe_dict[record[0]] = record[1]

            tmpdf = pd.DataFrame({'sharpe': sharpe_dict})
            tmpdf.index.name = 'code'
            tmpdf.to_csv(datapath('other_indicator_' + end_date + '.csv'))

            fund_datas.append(['SP500.SPI', 'GLNC', 'HSCI.HI'])
            fund_dates.append(dates[i])



    result_df = pd.DataFrame(result_datas, index=result_dates,
                             columns=['SP500.SPI', 'GLNC', 'HSCI.HI'])

    result_df.to_csv(datapath('otherlabelasset.csv'))


    fund_df = pd.DataFrame(fund_datas, index=fund_dates, columns=['SP500.SPI', 'GLNC', 'HSCI.HI'])
    fund_df.index.name = 'date'
    allocationdata.other_fund_sharpe_df = fund_df

    fund_df.to_csv(datapath('other_fund.csv'))


    print 'other label asset done'

    return result_df



def labelasset(allocationdata):


    his_week = allocationdata.fund_measure_lookback
    interval = allocationdata.fund_measure_adjust_period

    #start_date = allocationdata.start_date
    #end_date   = allocationdata.end_date
    #indexdf = Data.index_value(start_date, end_date, '000300.SH')


    # indexdf = DBData.index_value(allocationdata.data_start_date, allocationdata.end_date)[[Const.hs300_code]]
    # dates = indexdf.pct_change().index
    dates = DBData.trade_date_index(allocationdata.data_start_date, allocationdata.end_date)
    dates = [e.date() for e in dates]


    #allfunddf = Data.funds()

    #allfunddf = allocationdata.stock_df
    #stock_df = stockLabelAsset(allocationdata, dates, his_week, interval)

    #bondindexdf = Data.bond_index_value(start_date, end_date, Const.csibondindex_code)
    #allbonddf   = Data.bonds()
    #bondindexdf = allocationdata.index_df[[Const.csibondindex_code]]
    #allbonddf   = allocationdata.bond_df
    #bond_df = bondLabelAsset(allocationdata, dates, his_week, interval)

    #allmoneydf  = Data.moneys()
    #print allmoneydf
    #allmoneydf  = allocationdata.money_df
    #money_df = moneyLabelAsset(allocationdata, dates, his_week, interval)

    #allotherdf  = Data.others()
    #allotherdf  = allocationdata.other_df
    other_df = otherLabelAsset(allocationdata, dates, his_week, interval)

    df = pd.concat([stock_df, bond_df, money_df, other_df], axis = 1, join_axes=[stock_df.index])

    df.index.name = 'date'

    #df = df.dropna()

    dfr = df
    values = []
    for col in dfr.columns:
        rs = dfr[col].values
        vs = [1]
        for i in range(1, len(rs)):
            r = rs[i]
            v = vs[-1] * ( 1.0 + r )
            vs.append(v)
        values.append(vs)

    alldf = pd.DataFrame(np.matrix(values).T, index = dfr.index, columns = dfr.columns)

    allocationdata.label_asset_df = alldf
    alldf.to_csv(datapath('labelasset.csv'))

    allocationdata.label_asset_df = df


if __name__ == '__main__':
    start_date = '2007-01-05'
    end_date = '2016-04-22'

    indexdf = data.index_value(start_date, end_date, '000300.SH')
    dates = indexdf.pct_change().index

    allfunddf = data.funds()

    #his_week = 156
    interval = 26

    stock_df = stockLabelAsset(dates, interval, allfunddf, indexdf)

    bondindexdf = data.bond_index_value(start_date, end_date, const.csibondindex_code)
    allbonddf   = data.bonds()
    bond_df = bondLabelAsset(dates, interval, allbonddf, bondindexdf)

    allmoneydf  = data.moneys()
    #print allmoneydf
    money_df = moneyLabelAsset(dates, interval, allmoneydf, None)

    allotherdf  = data.others()
    other_df = otherLabelAsset(dates, interval, allotherdf, None)

    df = pd.concat([stock_df, bond_df, money_df, other_df], axis = 1, join_axes=[stock_df.index])

    df.to_csv(datapath('labelasset.csv'))

