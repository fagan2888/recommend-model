#coding=utf8


import numpy as np
import string
import sys
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


def stockLabelAsset(allocationdata, dates, his_week, interval):


    #df       = Data.funds()
    #df       = allocationdata.stock_df
    #dfr      = df.pct_change().fillna(0.0)
    #funddfr  = funddf.pct_change().fillna(0.0)

    indexdf   = DBData.index_value(dates[0], dates[-1])
    #indexdfr  = indexdf.pct_change().fillna(0.0)

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



    for i in range(his_week, len(dates)):

        if (i - his_week) % interval == 0:

            start_date                    = dates[i - his_week].strftime('%Y-%m-%d')
            end_date                      = dates[i].strftime('%Y-%m-%d')
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


            codes, indicator     = FundFilter.stockfundfilter(allocationdata, label_stock_df, indexdf[[Const.hs300_code]])
            #print time.time()
            fund_pool, fund_tags = ST.tagstockfund(allocationdata, label_stock_df[codes], this_index_df)

            #print time.time()
            #print
            allocationdf   = DFUtil.get_date_df(label_stock_df[fund_pool], allocation_start_date, end_date)
            fund_code, tag = FundSelector.select_stock(allocationdf, fund_tags)


            allcodes    = label_stock_df.columns
            filtercodes = codes
            poolcodes   = fund_pool
            selectcodes = fund_code


            #print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']
            #print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']

            f = open('./tmp/stock_pool_codes_' + end_date +'.txt','w')
            for code in poolcodes:
                f.write(str(code) + "\n")
            f.close()


            fund_dates.append(end_date)
            fund_datas.append([tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']])



        d = dates[i]
        result_dates.append(d)
        result_datas.append([funddfr.loc[d, tag['largecap']], funddfr.loc[d, tag['smallcap']], funddfr.loc[d, tag['rise']], funddfr.loc[d, tag['oscillation']], funddfr.loc[d, tag['decline']], funddfr.loc[d, tag['growth']], funddfr.loc[d, tag['value']]] )
        print d.strftime('%Y-%m-%d'), funddfr.loc[d, tag['largecap']], funddfr.loc[d, tag['smallcap']], funddfr.loc[d, tag['rise']], funddfr.loc[d, tag['oscillation']], funddfr.loc[d, tag['decline']], funddfr.loc[d, tag['growth']], funddfr.loc[d, tag['value']]


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
    result_df.to_csv('./tmp/stocklabelasset.csv')

    select_df = pd.DataFrame(select_datas, index = result_dates, columns=['allcodes','filtercodes','poolcode','selectcode'])
    select_df.to_csv('./tmp/stockselectasset.csv')

    fund_df = pd.DataFrame(fund_datas , index = fund_dates, columns=['largecap', 'smallcap', 'rise', 'oscillation', 'decline', 'growth', 'value'])
    fund_df.index.name = 'date'
    fund_df.to_csv('./tmp/stock_fund.csv')


    allocationdata.stock_fund_df = fund_df

    print 'stock label asset done'

    return result_df



def bondLabelAsset(allocationdata, dates, his_week, interval):

    #df  = Data.bonds()
    #dfr = df.pct_change().fillna(0.0)


    #funddfr = funddf.pct_change().fillna(0.0)
    #indexdfr = indexdf.pct_change().fillna(0.0)


    indexdf   = DBData.index_value(dates[0], dates[-1])


    pre_ratebond        = ''
    pre_creditbond      = ''
    pre_convertiblebond = ''


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


            start_date = dates[i - 52].strftime('%Y-%m-%d')
            end_date = dates[i].strftime('%Y-%m-%d')
            allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')


            last_end_date = dates[-1]
                        if i + interval < len(dates):
                                 last_end_date                 = dates[i + interval].strftime('%Y-%m-%d')


                        bond_df = DBData.bond_fund_value(start_date, last_end_date)
                        label_bond_df = DFUtil.get_date_df(bond_df, start_date, end_date)
                        this_index_df = DFUtil.get_date_df(indexdf, start_date, end_date)
                        funddfr = bond_df.pct_change().fillna(0.0)



            codes, indicator     = FundFilter.bondfundfilter(allocationdata, label_bond_df, indexdf[[Const.csibondindex_code]])
            fund_pool, fund_tags = ST.tagbondfund(allocationdata, label_bond_df[codes] ,this_index_df)


            allocationdf   = DFUtil.get_date_df(label_bond_df[fund_pool], allocation_start_date, end_date)
            fund_code, tag = FundSelector.select_bond(allocationdf, fund_tags)

            allcodes    = label_bond_df.columns
            filtercodes = codes
            poolcodes   = fund_pool
            selectcodes = fund_code


            if not tag.has_key('ratebond'):
                tag['ratebond'] = pre_ratebond
            else:
                pre_ratebond    = tag['ratebond']
            if not tag.has_key('creditbond'):
                tag['creditbond'] = pre_creditbond
            else:
                pre_creditbond  = tag['creditbond']
            if not tag.has_key('convertiblebond'):
                tag['convertiblebond'] = pre_convertiblebond
            else:
                pre_convertiblebond = tag['convertiblebond']


            #print tag['ratebond'], tag['creditbond'], tag['convertiblebond']
            # print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']


            fund_dates.append(end_date)
            fund_datas.append([tag['ratebond'] , tag['creditbond'], tag['convertiblebond']])


            f = open('./tmp/bond_pool_codes_' + end_date + '.txt','w')
            for code in poolcodes:
                f.write(str(code) + "\n")
            f.close()


        d = dates[i]
        result_dates.append(d)
        result_datas.append(
            [funddfr.loc[d, tag['ratebond']], funddfr.loc[d, tag['creditbond']], funddfr.loc[d, tag['convertiblebond']]])
        print d.strftime('%Y-%m-%d'), funddfr.loc[d, tag['ratebond']], funddfr.loc[d, tag['creditbond']], funddfr.loc[d, tag['convertiblebond']]


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
    result_df.to_csv('./tmp/bondlabelasset.csv')

    select_df = pd.DataFrame(select_datas, index = result_dates, columns=['allcodes','filtercodes','poolcode','selectcode'])
    select_df.to_csv('./tmp/bondselectasset.csv')


    fund_df = pd.DataFrame(fund_datas , index = fund_dates, columns=['ratebond', 'creditbond','convertiblebond'])
    fund_df.index.name = 'date'
    fund_df.to_csv('./tmp/bond_fund.csv')

    
    allocationdata.bond_fund_df = fund_df


    print 'bond label asset done'

    return result_df


def moneyLabelAsset(allocationdata, dates, his_week, interval):

    #funddfr = funddf.pct_change().fillna(0.0)

    tag = {}
    result_dates = []
    columns = []
    result_datas = []

    for i in range(his_week, len(dates)):

        if (i - his_week) % interval == 0:

            #start_date = dates[i - 52].strftime('%Y-%m-%d')
            #end_date = dates[i].strftime('%Y-%m-%d')
            #allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')

            #allocation_funddf = Data.money_value(allocation_start_date, end_date)
            #fund_codes, tag   = FundSelector.select_money(allocation_funddf)


            start_date = dates[i - 52].strftime('%Y-%m-%d')
            end_date = dates[i].strftime('%Y-%m-%d')
            allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')


            last_end_date = dates[-1]
                        if i + interval < len(dates):
                                 last_end_date                 = dates[i + interval].strftime('%Y-%m-%d')


                        money_df = DBData.money_fund_value(start_date, last_end_date)
                        label_money_df = DFUtil.get_date_df(money_df, start_date, end_date)
                        #this_index_df = DFUtil.get_date_df(indexdf, start_date, end_date)
                        funddfr = money_df.pct_change().fillna(0.0)


            allocationdf   = DFUtil.get_date_df(label_money_df, allocation_start_date, end_date)
            fund_codes, tag   = FundSelector.select_money(allocationdf)

        #print tag
        # print tag
        # print fund_codes

        d = dates[i]
        result_dates.append(d)
        result_datas.append(
            [funddfr.loc[d, tag['money']]])

        print d.strftime('%Y-%m-%d'), ',', funddfr.loc[d, tag['money']]

    result_df = pd.DataFrame(result_datas, index=result_dates,columns=['money'])
    result_df.to_csv('./tmp/moneylabelasset.csv')


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

    for i in range(his_week, len(dates)):

        d = dates[i]

        result_dates.append(d)
        result_datas.append(
            [funddfr.loc[d, 'SP500.SPI'], funddfr.loc[d, 'GLNC'], funddfr.loc[d, 'HSCI.HI']] )

        print d.strftime('%Y-%m-%d'), ',', funddfr.loc[d, 'SP500.SPI'], ',', funddfr.loc[d, 'GLNC'], ',' , funddfr.loc[d, 'HSCI.HI']

    result_df = pd.DataFrame(result_datas, index=result_dates,
                             columns=['SP500.SPI', 'GLNC', 'HSCI.HI'])

    result_df.to_csv('./tmp/otherlabelasset.csv')

    print 'other label asset done'

    return result_df


def labelasset(allocationdata):


    his_week = allocationdata.fund_measure_lookback
    interval = allocationdata.fund_measure_adjust_period

    #start_date = allocationdata.start_date
    #end_date   = allocationdata.end_date
    #indexdf = Data.index_value(start_date, end_date, '000300.SH')


    indexdf = DBData.index_value(allocationdata.data_start_date, allocationdata.end_date)[[Const.hs300_code]]
    dates = indexdf.pct_change().index

    #allfunddf = Data.funds()

    #allfunddf = allocationdata.stock_df
    stock_df = stockLabelAsset(allocationdata, dates, his_week, interval)

    #bondindexdf = Data.bond_index_value(start_date, end_date, Const.csibondindex_code)
    #allbonddf   = Data.bonds()
    #bondindexdf = allocationdata.index_df[[Const.csibondindex_code]]
    #allbonddf   = allocationdata.bond_df
    bond_df = bondLabelAsset(allocationdata, dates, his_week, interval)

    #allmoneydf  = Data.moneys()
    #print allmoneydf
    #allmoneydf  = allocationdata.money_df
    money_df = moneyLabelAsset(allocationdata, dates, his_week, interval)

    #allotherdf  = Data.others()
    #allotherdf  = allocationdata.other_df
    other_df = otherLabelAsset(allocationdata, dates, his_week, interval)

    df = pd.concat([stock_df, bond_df, money_df, other_df], axis = 1, join_axes=[stock_df.index])

    df.index.name = 'date'

    df = df.dropna()

    allocationdata.label_asset_df = df

    df.to_csv('./tmp/labelasset.csv')


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

    df.to_csv('./tmp/labelasset.csv')
