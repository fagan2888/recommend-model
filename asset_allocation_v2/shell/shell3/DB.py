#coding=utf8


import os
import sys
sys.path.append('shell')
import pymysql
pymysql.install_as_MySQLdb()

# import MySQLdb
from . import config
import string
import pandas as pd
import numpy as np
from datetime import datetime
from . import FundIndicator
from . import AllocationData
from . import RiskPosition
import time
from .Const import datapath


def fund_measure(allocationdata):


    conn = MySQLdb.connect(**config.db_asset)
    cursor = conn.cursor()


    base_sql = 'replace into fund_measure (fm_start_date, fm_end_date, fm_adjust_period, fm_look_back, fm_fund_type, fm_fund_code, fm_jensen, fm_ppw, fm_stability, fm_sortino, fm_sharpe, fm_high_postion_prefer, fm_low_position_prefer, fm_largecap_prefer, fm_smallcap_prefer, fm_growth_prefer, fm_value_prefer, fm_largecap_fitness, fm_smallcap_fitness, fm_rise_fitness, fm_decline_fitness, fm_oscillation_fitness, fm_growth_fitness, fm_value_fitness, fm_ratebond, fm_creditbond, fm_convertiblebond, fm_sp500, fm_gold, fm_hs, created_at, updated_at) values ("%s","%s",%d, %d, "%s","%s", %f,%f,%f,%f,%f, %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d, "%s","%s")'


    stock_fund_measure    = allocationdata.stock_fund_measure
    stock_fund_label      = allocationdata.stock_fund_label
    bond_fund_measure     = allocationdata.bond_fund_measure
    bond_fund_label       = allocationdata.bond_fund_label
    money_fund_sharpe_df  = allocationdata.money_fund_sharpe_df
    other_fund_sharpe_df  = allocationdata.other_fund_sharpe_df


    dates = list(stock_fund_measure.keys())
    dates.sort()

    for date in dates:

        stock_measure_df = stock_fund_measure[date]
        stock_label_df   = stock_fund_label[date]

        for code in stock_fund_measure[date].index:

            #if code == '240002':
            #    print date

            jensen                 = 0
            ppw                    = 0
            sortino                = 0
            stability              = 0
            sharpe                 = 0
            high_position_prefer   = 0
            low_position_prefer    = 0
            largecap_prefer        = 0
            smallcap_prefer        = 0
            growth_prefer          = 0
            value_prefer           = 0
            largecap_fitness       = 0
            smallcap_fitness       = 0
            rise_fitness           = 0
            decline_fitness        = 0
            oscillation_fitness    = 0
            growth_fitness         = 0
            value_fitness          = 0
            ratebond               = 0
            convertiblebond        = 0
            sp500                  = 0
            gold                   = 0
            hs                     = 0


            if not np.isnan(stock_measure_df.loc[code,'jensen']):
                jensen = stock_measure_df.loc[code,'jensen']
            if not np.isnan(stock_measure_df.loc[code,'ppw']):
                ppw = stock_measure_df.loc[code,'ppw']
            if not np.isnan(stock_measure_df.loc[code,'sortino']):
                sortino = stock_measure_df.loc[code,'sortino']
            if not np.isnan(stock_measure_df.loc[code,'stability']):
                stability = stock_measure_df.loc[code,'stability']
            if not np.isnan(stock_measure_df.loc[code,'sharpe']):
                sharpe = stock_measure_df.loc[code,'sharpe']


            if code in set(stock_label_df.index):
                if not np.isnan(stock_label_df.loc[code,'high_position_prefer']):
                    high_position_prefer = stock_label_df.loc[code,'high_position_prefer']
                if not np.isnan(stock_label_df.loc[code,'low_position_prefer']):
                    low_position_prefer = stock_label_df.loc[code,'low_position_prefer']
                if not np.isnan(stock_label_df.loc[code,'largecap_prefer']):
                    largecap_prefer = stock_label_df.loc[code,'largecap_prefer']
                if not np.isnan(stock_label_df.loc[code,'smallcap_prefer']):
                    smallcap_prefer = stock_label_df.loc[code,'smallcap_prefer']
                if not np.isnan(stock_label_df.loc[code,'growth_prefer']):
                    growth_prefer = stock_label_df.loc[code,'growth_prefer']
                if not np.isnan(stock_label_df.loc[code,'value_prefer']):
                    value_prefer = stock_label_df.loc[code,'value_prefer']
                if not np.isnan(stock_label_df.loc[code,'largecap_fitness']):
                    largecap_fitness = stock_label_df.loc[code,'largecap_fitness']
                if not np.isnan(stock_label_df.loc[code,'smallcap_fitness']):
                    smallcap_fitness = stock_label_df.loc[code,'smallcap_fitness']
                if not np.isnan(stock_label_df.loc[code,'rise_fitness']):
                    rise_fitness = stock_label_df.loc[code,'rise_fitness']
                if not np.isnan(stock_label_df.loc[code,'decline_fitness']):
                    decline_fitness = stock_label_df.loc[code,'decline_fitness']
                if not np.isnan(stock_label_df.loc[code,'oscillation_fitness']):
                    oscillation_fitness = stock_label_df.loc[code,'oscillation_fitness']
                if not np.isnan(stock_label_df.loc[code,'growth_fitness']):
                    growth_fitness = stock_label_df.loc[code,'growth_fitness']
                if not np.isnan(stock_label_df.loc[code,'value_fitness']):
                    value_fitness = stock_label_df.loc[code,'value_fitness']


            sql = base_sql % (allocationdata.start_date, date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback,'stock' ,code , jensen, ppw, stability, sortino, sharpe, high_position_prefer, low_position_prefer, largecap_prefer, smallcap_prefer, growth_prefer, value_prefer, largecap_fitness, smallcap_fitness, rise_fitness, decline_fitness, oscillation_fitness, growth_fitness, value_fitness, 0, 0, 0, 0, 0, 0 ,datetime.now(), datetime.now())

            cursor.execute(sql)


    dates = list(bond_fund_measure.keys())
    dates.sort()


    for date in dates:

        bond_measure_df  = bond_fund_measure[date]
        bond_label_df    = bond_fund_label[date]

        for code in bond_fund_measure[date].index:

            jensen                 = 0
            ppw                    = 0
            sortino                = 0
            stability              = 0
            sharpe                 = 0
            ratebond               = 0
            creditbond             = 0
            convertiblebond        = 0


            if not np.isnan(bond_measure_df.loc[code,'jensen']):
                jensen = bond_measure_df.loc[code,'jensen']
            if not np.isnan(bond_measure_df.loc[code,'ppw']):
                ppw = bond_measure_df.loc[code,'ppw']
            if not np.isnan(bond_measure_df.loc[code,'sortino']):
                sortino = bond_measure_df.loc[code,'sortino']
            if not np.isnan(bond_measure_df.loc[code,'stability']):
                stability = bond_measure_df.loc[code,'stability']
            if not np.isnan(bond_measure_df.loc[code,'sharpe']):
                sharpe = bond_measure_df.loc[code,'sharpe']

            if code in set(bond_label_df.index):
                if not np.isnan(bond_label_df.loc[code,'ratebond']):
                    ratebond = bond_label_df.loc[code,'ratebond']
                if not np.isnan(bond_label_df.loc[code,'creditbond']):
                    creditbond = bond_label_df.loc[code,'creditbond']
                #if not np.isnan(bond_label_df.loc[code,'convertiblebond']):
                #    convertiblebond = bond_label_df.loc[code,'convertiblebond']


            sql = base_sql % (allocationdata.start_date, date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback,'bond' ,code , jensen, ppw, stability, sortino, sharpe, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ratebond, creditbond, convertiblebond, 0, 0, 0 ,datetime.now(), datetime.now())

            #print sql
            cursor.execute(sql)


    dates = list(money_fund_sharpe_df.index)
    dates.sort()

    for date in dates:
        money_sharpe = money_fund_sharpe_df.loc[date, 'money']
        sql = base_sql % (allocationdata.start_date, date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback,'money' ,'213009' , 0, 0, 0, 0, money_sharpe, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,datetime.now(), datetime.now())
        cursor.execute(sql)


    dates = list(other_fund_sharpe_df.index)
    dates.sort()
    for date in dates:

        SP500_sharpe = other_fund_sharpe_df.loc[date,'SP500.SPI']
        GLNC_sharpe  = other_fund_sharpe_df.loc[date,'GLNC']
        HSCI_sharpe = other_fund_sharpe_df.loc[date,'HSCI.HI']

        sql = base_sql % (allocationdata.start_date, date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback,'other' ,'513500' , 0, 0, 0, 0, SP500_sharpe , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,datetime.now(), datetime.now())
        cursor.execute(sql)

        sql = base_sql % (allocationdata.start_date, date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback,'other' ,'159934' , 0, 0, 0, 0, GLNC_sharpe , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,datetime.now(), datetime.now())
        cursor.execute(sql)

        sql = base_sql % (allocationdata.start_date, date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback,'other' ,'513600' , 0, 0, 0, 0, HSCI_sharpe , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,datetime.now(), datetime.now())
        cursor.execute(sql)


    conn.commit()
    conn.close()


    print('fund measure done')



def label_asset(allocationdata):


    conn = MySQLdb.connect(**config.db_asset)
    cursor = conn.cursor()


    label_asset_df            = allocationdata.label_asset_df
    stock_fund_df             = allocationdata.stock_fund_df
    bond_fund_df              = allocationdata.bond_fund_df
    equal_risk_asset_ratio_df = allocationdata.equal_risk_asset_ratio_df
    equal_risk_asset_df       = allocationdata.equal_risk_asset_df


    dates = equal_risk_asset_df.index.values
    dates.sort()
    d = dates[0]


    label_asset_df = label_asset_df[label_asset_df.index >= d]
    dates = label_asset_df.index.values
    dates.sort()


    columns = label_asset_df.columns
    values = []
    for col in columns:
        vs = [1]
        for i in range(1, len(dates)):
            r = label_asset_df.loc[dates[i], col]
            v = vs[-1] * ( 1 + r)
            vs.append(v)
        values.append(vs)


    m = np.matrix(values)
    label_asset_df = pd.DataFrame(m.T, index = equal_risk_asset_df.index.values, columns = columns)

    label_asset_df.to_csv(datapath('test.csv'))


    base_sql = 'replace into fixed_risk_asset (fra_start_date, fra_adjust_period, fra_asset_look_back, fra_risk_adjust_period, fra_risk_look_back, fra_fund_type, fra_fund_code, fra_asset_type, fra_position, fra_asset_label, fra_net_value, fra_date, fra_annual_return, fra_sharpe, fra_std, fra_maxdrawdown ,created_at, updated_at) values ("%s", %d, %d, %d, %d, "%s","%s", "%s" ,%f, "%s", %f, "%s", %f, %f, %f, %f, "%s", "%s")'



    stock_tags        = ['largecap','smallcap','rise','oscillation','decline','growth','value', 'SP500.SPI','GLNC','HSCI.HI']
    #origin_bond_tags  = ['ratebond','creditbond','convertiblebond']
    origin_bond_tags  = ['ratebond','creditbond']
    #bond_tags         = ['convertiblebond']
    money_tags        = ['money']
    other_tags        = ['SP500.SPI','GLNC','HSCI.HI']


    stock_fund_df_dates = set(stock_fund_df.index.values)
    equal_risk_asset_ratio_dates = set(equal_risk_asset_ratio_df.index.values)


    sfdd = []
    for d in stock_fund_df_dates:
        sfdd.append(datetime.strptime(d, '%Y-%m-%d').date())
    sfdd.sort()


    for label in stock_tags:

        fund = ''
        dates = label_asset_df.index.values
        dates.sort()
        start_date = dates[0]


        for i in range(0 ,len(sfdd)):
            if sfdd[i] > start_date and (label not in set(other_tags)) :
                fund = stock_fund_df.loc[sfdd[i-1].strftime('%Y-%m-%d')    ,label]
                break

        navs = []

        for d in dates:

            if (d.strftime('%Y-%m-%d') in stock_fund_df_dates) and (label not in set(other_tags)):
                fund = stock_fund_df.loc[d.strftime('%Y-%m-%d'), label]
            if label in set(other_tags):
                fund = label

            net_value = label_asset_df.loc[d, label]

            navs.append(net_value)

            sharpe = FundIndicator.portfolio_sharpe(navs)
            returns= FundIndicator.portfolio_return(navs)
            risk   = FundIndicator.portfolio_risk(navs)
            maxdrawdown = FundIndicator.portfolio_maxdrawdown(navs)


            if np.isnan(sharpe) or np.isinf(sharpe):
                sharpe = 0
            if np.isnan(returns) or np.isinf(returns):
                returns = 0
            if np.isnan(risk) or np.isnan(risk):
                risk = 0
            if np.isnan(maxdrawdown) or np.isnan(maxdrawdown):
                maxdrawdown = 0

            #print base_sql
            #print fund
            if label in set(other_tags):
                if label == 'SP500.SPI':
                    fund = '513500'
                if label == 'HSCI.HI':
                    fund = '513600'
                if label == 'GLNC':
                    fund = '518880'
                sql = base_sql % (allocationdata.start_date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback, allocationdata.fixed_risk_asset_risk_adjust_period, allocationdata.fixed_risk_asset_risk_lookback, 'other', fund, 'origin', 1.0, label, net_value, d, returns, sharpe, risk, maxdrawdown , datetime.now(), datetime.now())
            else:
                sql = base_sql % (allocationdata.start_date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback, allocationdata.fixed_risk_asset_risk_adjust_period, allocationdata.fixed_risk_asset_risk_lookback, 'stock', fund, 'origin', 1.0, label, net_value, d, returns, sharpe, risk, maxdrawdown , datetime.now(), datetime.now())
            #print net_value
            cursor.execute(sql)



    for label in stock_tags:


        fund = ''
        position = 0.0
        dates = equal_risk_asset_df.index.values
        dates.sort()
        start_date = dates[0]

        for i in range(0 ,len(sfdd)):
            if (sfdd[i] > start_date) and (label not in set(other_tags)):
                fund = stock_fund_df.loc[sfdd[i-1].strftime('%Y-%m-%d')    ,label]
                break

        navs = []

        for d in dates:

            if (d.strftime('%Y-%m-%d') in stock_fund_df_dates) and (label not in set(other_tags)):
                fund = stock_fund_df.loc[d.strftime('%Y-%m-%d'), label]
            if d in equal_risk_asset_ratio_dates:
                position = equal_risk_asset_ratio_df.loc[d, label]

            net_value = equal_risk_asset_df.loc[d, label]

            navs.append(net_value)

            sharpe = FundIndicator.portfolio_sharpe(navs)
            returns= FundIndicator.portfolio_return(navs)
            risk   = FundIndicator.portfolio_risk(navs)
            maxdrawdown = FundIndicator.portfolio_maxdrawdown(navs)


            if np.isnan(sharpe) or np.isinf(sharpe):
                sharpe = 0
            if np.isnan(returns) or np.isinf(returns):
                returns = 0
            if np.isnan(risk) or np.isnan(risk):
                risk = 0
            if np.isnan(maxdrawdown) or np.isnan(maxdrawdown):
                maxdrawdown = 0


            #print fund
            #print base_sql
            if label in set(other_tags):
                if label == 'SP500.SPI':
                    fund = '513500'
                if label == 'HSCI.HI':
                    fund = '513600'
                if label == 'GLNC':
                    fund = '518880'
                sql = base_sql % (allocationdata.start_date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback, allocationdata.fixed_risk_asset_risk_adjust_period, allocationdata.fixed_risk_asset_risk_lookback, 'other', fund, 'fixed_risk', position, label, net_value, d, returns, sharpe, risk, maxdrawdown , datetime.now(), datetime.now())
            else:
                sql = base_sql % (allocationdata.start_date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback, allocationdata.fixed_risk_asset_risk_adjust_period, allocationdata.fixed_risk_asset_risk_lookback, 'stock', fund, 'fixed_risk', position, label, net_value, d, returns, sharpe, risk, maxdrawdown , datetime.now(), datetime.now())

            #print net_value
            #print sql
            cursor.execute(sql)


    bond_fund_df_dates = set(bond_fund_df.index.values)
    dfdd = []
    for d in bond_fund_df_dates:
        dfdd.append(datetime.strptime(d, '%Y-%m-%d').date())
    dfdd.sort()


    for label in origin_bond_tags:

        fund = ''
        dates = label_asset_df.index.values
        dates.sort()

        start_date = dates[0]
        for i in range(0 ,len(dfdd)):
            if dfdd[i] > start_date:
                fund = bond_fund_df.loc[dfdd[i-1].strftime('%Y-%m-%d'),label]
                break

        navs = []

        for d in dates:

            if d.strftime('%Y-%m-%d') in bond_fund_df_dates:
                fund = bond_fund_df.loc[d.strftime('%Y-%m-%d'), label]

            net_value = label_asset_df.loc[d, label]

            navs.append(net_value)

            sharpe = FundIndicator.portfolio_sharpe(navs)
            returns= FundIndicator.portfolio_return(navs)
            risk   = FundIndicator.portfolio_risk(navs)
            maxdrawdown = FundIndicator.portfolio_maxdrawdown(navs)

            if np.isnan(sharpe) or np.isinf(sharpe):
                sharpe = 0
            if np.isnan(returns) or np.isinf(returns):
                returns = 0
            if np.isnan(risk) or np.isnan(risk):
                risk = 0
            if np.isnan(maxdrawdown) or np.isnan(maxdrawdown):
                maxdrawdown = 0

            #print fund
            #print base_sql
            sql = base_sql % (allocationdata.start_date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback, allocationdata.fixed_risk_asset_risk_adjust_period, allocationdata.fixed_risk_asset_risk_lookback, 'bond', fund, 'origin', 1.0, label, net_value, d, returns, sharpe, risk, maxdrawdown, datetime.now(), datetime.now())

            #print net_value
            #print sql
            cursor.execute(sql)



    bond_fund_df_dates = set(bond_fund_df.index.values)
    dfdd = []
    for d in bond_fund_df_dates:
        dfdd.append(datetime.strptime(d, '%Y-%m-%d').date())
    dfdd.sort()


    for label in origin_bond_tags:

        fund = ''
        dates = label_asset_df.index.values
        dates.sort()

        start_date = dates[0]
        for i in range(0 ,len(dfdd)):
            if dfdd[i] > start_date:
                fund = bond_fund_df.loc[dfdd[i-1].strftime('%Y-%m-%d'),label]
                break

        navs = []

        for d in dates:

            if d.strftime('%Y-%m-%d') in bond_fund_df_dates:
                fund = bond_fund_df.loc[d.strftime('%Y-%m-%d'), label]

            net_value = label_asset_df.loc[d, label]

            navs.append(net_value)


            sharpe = FundIndicator.portfolio_sharpe(navs)
            returns= FundIndicator.portfolio_return(navs)
            risk   = FundIndicator.portfolio_risk(navs)
            maxdrawdown = FundIndicator.portfolio_maxdrawdown(navs)

            if np.isnan(sharpe) or np.isinf(sharpe):
                sharpe = 0
            if np.isnan(returns) or np.isinf(returns):
                returns = 0
            if np.isnan(risk) or np.isnan(risk):
                risk = 0
            if np.isnan(maxdrawdown) or np.isnan(maxdrawdown):
                maxdrawdown = 0

            #print fund
            #print base_sql
            sql = base_sql % (allocationdata.start_date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback, allocationdata.fixed_risk_asset_risk_adjust_period, allocationdata.fixed_risk_asset_risk_lookback, 'bond', fund, 'origin', 1.0, label, net_value, d, returns, sharpe, risk, maxdrawdown, datetime.now(), datetime.now())

            #print net_value
            #print sql
            cursor.execute(sql)


    '''

    for label in bond_tags:

        fund = ''
        position = 0.0
        dates = label_asset_df.index.values
        dates.sort()

        navs = []

        for d in dates:

            if d in bond_fund_df_dates:
                fund = bond_fund_df.loc[d, label]
            if d in equal_risk_asset_ratio_dates:
                position = equal_risk_asset_ratio_df.loc[d, label]
            net_value = equal_risk_asset_df.loc[d, label]

            navs.append(net_value)


            sharpe = FundIndicator.portfolio_sharpe(navs)
            returns= FundIndicator.portfolio_return(navs)
            risk   = FundIndicator.portfolio_risk(navs)
            maxdrawdown = FundIndicator.portfolio_maxdrawdown(navs)

            if np.isnan(sharpe) or np.isinf(sharpe):
                sharpe = 0
            if np.isnan(returns) or np.isinf(returns):
                returns = 0
            if np.isnan(risk) or np.isnan(risk):
                risk = 0
            if np.isnan(maxdrawdown) or np.isnan(maxdrawdown):
                maxdrawdown = 0

            #print base_sql


            #print base_sql
            sql = base_sql % (allocationdata.start_date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback, allocationdata.fixed_risk_asset_risk_adjust_period, allocationdata.fixed_risk_asset_risk_lookback, 'bond', fund, 'fixed_risk', position, label, net_value, d, returns, sharpe, risk, maxdrawdown, datetime.now(), datetime.now())


            #print sql
            cursor.execute(sql)
    '''

    conn.commit()
    conn.close()


    print('label asset done')



def asset_allocation(allocationdata):


    conn = MySQLdb.connect(**config.db_asset)
    cursor = conn.cursor()


    base_sql = "replace into asset_allocation (aa_start_date, aa_date, aa_look_back, aa_adjust_period, aa_net_value, aa_largecap_ratio, aa_smallcap_ratio, aa_rise_ratio, aa_oscillation_ratio, aa_decline_ratio, aa_growth_ratio, aa_value_ratio, aa_convertible_bond_ratio, aa_rate_bond_ratio ,aa_creditable_ratio, aa_sp500_ratio, aa_gold_ratio, aa_hs_ratio, aa_highrisk_ratio, aa_lowrisk_ratio, aa_sharpe, aa_annual_return, aa_std, aa_maxdrawdown, aa_asset_type, created_at, updated_at) values ('%s', '%s', %d, %d, %f, %f, %f, %f, %f, %f, %f,%f, %f, %f, %f, %f, %f, %f, %f,%f, %f, %f, %f, %f,'%s',  '%s', '%s')"


    high_risk_position_df    = allocationdata.high_risk_position_df
    low_risk_position_df     = allocationdata.low_risk_position_df
    highlow_risk_position_df = allocationdata.highlow_risk_position_df
    high_risk_asset_df       = allocationdata.high_risk_asset_df
    low_risk_asset_df        = allocationdata.low_risk_asset_df
    highlow_risk_asset_df    = allocationdata.highlow_risk_asset_df


    largecap_ratio         = 0.0
    smallcap_ratio         = 0.0
    rise_ratio             = 0.0
    oscillation_ratio      = 0.0
    decline_ratio          = 0.0
    growth_ratio           = 0.0
    value_ratio            = 0.0
    convertible_ratio      = 0.0
    rate_bond_ratio        = 0.0
    credit_bond_ratio      = 0.0
    SP500_SPI_ratio        = 0.0
    SPGSGCTR_SPI_ratio     = 0.0
    HSCI_HI_ratio          = 0.0
    highrisk_ratio         = 0.0
    lowrisk_ratio          = 0.0


    navs = []

    dates = high_risk_asset_df.index.values
    dates.sort()

    #print dates
    high_position_dates = set(high_risk_position_df.index.values)
    #print high_position_dates

    #print high_risk_position_df

    for d in dates:
        #print high_risk_asset_df
        net_value = high_risk_asset_df.loc[d, 'high_risk_asset']
        navs.append(net_value)

        str_d = d.strftime('%Y-%m-%d')
        if str_d in high_position_dates:
            largecap_ratio = high_risk_position_df.loc[str_d, 'largecap']
            smallcap_ratio = high_risk_position_df.loc[str_d, 'smallcap']
            rise_ratio = high_risk_position_df.loc[str_d, 'rise']
            oscillation_ratio = high_risk_position_df.loc[str_d, 'oscillation']
            decline_ratio = high_risk_position_df.loc[str_d, 'decline']
            growth_ratio = high_risk_position_df.loc[str_d, 'growth']
            value_ratio = high_risk_position_df.loc[str_d, 'value']
            #convertible_ratio = high_risk_position_df.loc[d, 'convertiblebond']
            SP500_SPI_ratio = high_risk_position_df.loc[str_d, 'SP500.SPI']
            SPGSGCTR_SPI_ratio = high_risk_position_df.loc[str_d, 'GLNC']
            HSCI_HI_ratio = high_risk_position_df.loc[str_d, 'HSCI.HI']


        #print base_sql


        sharpe = FundIndicator.portfolio_sharpe(navs)
        returns= FundIndicator.portfolio_return(navs)
        risk   = FundIndicator.portfolio_risk(navs)
        maxdrawdown = FundIndicator.portfolio_maxdrawdown(navs)


        if np.isnan(sharpe) or np.isinf(sharpe):
            sharpe = 0
        if np.isnan(returns) or np.isinf(returns):
            returns = 0
        if np.isnan(risk) or np.isnan(risk):
            risk = 0
        if np.isnan(maxdrawdown) or np.isnan(maxdrawdown):
            maxdrawdown = 0

        sql = base_sql % (allocationdata.start_date, d, allocationdata.allocation_lookback, allocationdata.allocation_adjust_period, net_value, largecap_ratio, smallcap_ratio, rise_ratio, oscillation_ratio, decline_ratio, growth_ratio, value_ratio, convertible_ratio, 0.0, 0.0, SP500_SPI_ratio, SPGSGCTR_SPI_ratio, HSCI_HI_ratio, 0.0, 0.0, sharpe, returns, risk, maxdrawdown, 'highrisk', datetime.now(), datetime.now())

        #print sql
        cursor.execute(sql)


    navs = []
    dates = low_risk_asset_df.index.values
    dates.sort()
    low_position_dates = set(low_risk_position_df.index.values)


    #print high_risk_position_df


    for d in dates:
        #print high_risk_asset_df
        net_value = low_risk_asset_df.loc[d, 'low_risk_asset']
        navs.append(net_value)
        str_d = d.strftime('%Y-%m-%d')
        if str_d in low_position_dates:
            rate_bond_ratio   = low_risk_position_df.loc[str_d, 'ratebond']
            credit_bond_ratio = low_risk_position_df.loc[str_d, 'creditbond']


        #print base_sql


        sharpe = FundIndicator.portfolio_sharpe(navs)
        returns= FundIndicator.portfolio_return(navs)
        risk   = FundIndicator.portfolio_risk(navs)
        maxdrawdown = FundIndicator.portfolio_maxdrawdown(navs)


        if np.isnan(sharpe) or np.isinf(sharpe):
            sharpe = 0
        if np.isnan(returns) or np.isinf(returns):
            returns = 0
        if np.isnan(risk) or np.isnan(risk):
            risk = 0
        if np.isnan(maxdrawdown) or np.isnan(maxdrawdown):
            maxdrawdown = 0

        sql = base_sql % (allocationdata.start_date, d, allocationdata.allocation_lookback, allocationdata.allocation_adjust_period, net_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rate_bond_ratio, credit_bond_ratio, 0.0, 0.0, 0.0, 0.0, 0.0, sharpe, returns, risk, maxdrawdown, 'lowrisk', datetime.now(), datetime.now())

        #print sql
        cursor.execute(sql)


    navs = []
    dates = highlow_risk_asset_df.index.values
    dates.sort()
    highlow_position_dates = set(highlow_risk_position_df.index.values)




    #print high_risk_position_df

    for d in dates:

        #print high_risk_asset_df
        net_value = highlow_risk_asset_df.loc[d, 'highlow_risk_asset']
        navs.append(net_value)
        str_d = d.strftime('%Y-%m-%d')


        if str_d in highlow_position_dates:
            #print 'hehe'
            highrisk_ratio   =  highlow_risk_position_df.loc[str_d, 'high_risk_asset']
            lowrisk_ratio    =  highlow_risk_position_df.loc[str_d, 'low_risk_asset']

        #print base_sql


        sharpe = FundIndicator.portfolio_sharpe(navs)
        returns= FundIndicator.portfolio_return(navs)
        risk   = FundIndicator.portfolio_risk(navs)
        maxdrawdown = FundIndicator.portfolio_maxdrawdown(navs)


        if np.isnan(sharpe) or np.isinf(sharpe):
            sharpe = 0
        if np.isnan(returns) or np.isinf(returns):
            returns = 0
        if np.isnan(risk) or np.isnan(risk):
            risk = 0
        if np.isnan(maxdrawdown) or np.isnan(maxdrawdown):
            maxdrawdown = 0
        sql = base_sql % (allocationdata.start_date, d, allocationdata.allocation_lookback, allocationdata.allocation_adjust_period, net_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, highrisk_ratio, lowrisk_ratio, sharpe, returns, risk, maxdrawdown, 'highlowrisk', datetime.now(), datetime.now())

        cursor.execute(sql)

    conn.commit()
    conn.close()

    print('asset allocation done')

    return 0



def risk_allocation_list(risk_value, risk_begin_date):

    conn = MySQLdb.connect(**config.db_asset)
    cursor = conn.cursor()


    base_sql = "replace into risk_asset_allocation_list (ra_risk, ra_date) values (%f, %s)"
    sql = base_sql % (risk_value, risk_begin_date)
    # cursor.execute(sql)
    conn.commit()
    conn.close()
    print("insert one row into risk_allocation_list")



def risk_allocation_ratio(df, lid):
    conn = MySQLdb.connect(**config.db_asset)
    cursor = conn.cursor()
    base_sql = "replace into risk_asset_allocation_list (ra_alloc_id, ra_transfer_date, ra_fund_code, ra_fund_ratio) values (%d, %s, %s, %f)"

    dates = table.index.unique()

    for date in dates:
        funds_ratio_day = table.loc[oneDay, ['fund', 'ratio']]
        row = operates.get(['fund', 'ratio'])
        values = row.values
        ites = values.size / 2
        for ite in range(ites):
            if ites == 1:
                value = values
            else:
                value = values[ite]
            fcode = value[0]
            fratio = value[1]
            sql = base_sql % (lid, date, fcode, fratio)
            # cursor.execute(sql)
    conn.commit()
    conn.close()
    print("insert one row into risk_allocation_list")



def riskhighlowriskasset(allocationdata):

    start_date = '2010-01-01'

    df = pd.read_csv(datapath('risk_portfolio.csv'), index_col = 'date', parse_dates = ['date'])

    conn = MySQLdb.connect(**config.db_asset)
    cursor = conn.cursor()

    for risk in df.columns:
        col = risk
        risk = string.atoi(risk) / 10.0

        sql = "replace into risk_asset_allocation_list (ra_risk, ra_date, created_at, updated_at) values (%f, '%s', '%s', '%s')" % (risk, start_date, datetime.now(), datetime.now())
        cursor.execute(sql)
        sql = 'select id from risk_asset_allocation_list where ra_risk = %f' % (risk)
        cursor.execute(sql)
        record = cursor.fetchone()
        risk_id = record[0]

        for date in df.index:
            # sql = "replace into risk_asset_allocation_nav (ra_alloc_id, ra_date, ra_nav, created_at, updated_at) values (%d, '%s', %f, '%s', '%s')" % (risk_id, date, df.loc[date, col], datetime.now(), datetime.now())
            # print sql
            cursor.execute(sql)

    all_code_position = RiskPosition.risk_position()

    allriskposition      = {}

    for record in all_code_position:
        risk_rank = record[0]
        date      = record[1]
        code      = record[2]
        f_id      = allocationdata.fund_code_id_dict[code]
        ratio     = record[3]

        dateposition = allriskposition.setdefault(risk_rank, {})
        ps           = dateposition.setdefault(date,{})
        ps[f_id]     = ratio



    #dates = list(allriskposition.keys())
    #dates.sort()
    #ps = allriskpositions[dates[0]]


    for risk_rank in list(allriskposition.keys()):

        sql       = 'select id from risk_asset_allocation_list where ra_risk = %f' % (risk_rank)
        cursor.execute(sql)
        record    = cursor.fetchone()
        list_id   = record[0]

        dateposition = allriskposition[risk_rank]
        dates = list(dateposition.keys())
        dates.sort()
        ps = dateposition[dates[0]]
        last_date = dates[0]

        for i in range(1, len(dates)):

            d = dates[i]
            current_ps = dateposition[d]
            if not current_ps == ps:
                current_ratio = 0.0
                for f_id in list(ps.keys()):
                    ratio = ps[f_id]
                    current_ratio += ratio
                    sql       = 'replace into risk_asset_allocation_ratio (ra_alloc_id, ra_transfer_date, ra_fund_id, ra_fund_ratio,  created_at, updated_at) values (%d, "%s", %d, %f, "%s", "%s")' % (list_id, last_date, f_id, ratio, datetime.now(), datetime.now())
                    #print sql
                    #cursor.execute(sql)
                money_ratio = 1.0 - current_ratio
                if money_ratio > 0.00000099:
                    sql       = 'replace into risk_asset_allocation_ratio (ra_alloc_id, ra_transfer_date, ra_fund_id, ra_fund_ratio,  created_at, updated_at) values (%d, "%s", %d, %f, "%s", "%s")' % (list_id, last_date, 30003446, money_ratio, datetime.now(), datetime.now())
                    #cursor.execute(sql)
                ps = current_ps
                last_date = d

        current_ratio = 0.0
        for f_id in list(ps.keys()):
            ratio = ps[f_id]
            current_ratio += ratio
            sql       = 'replace into risk_asset_allocation_ratio (ra_alloc_id, ra_transfer_date, ra_fund_id, ra_fund_ratio,  created_at, updated_at) values (%d, "%s", %d, %f, "%s", "%s")' % (list_id, last_date, f_id, ratio, datetime.now(), datetime.now())
            #print sql
            #cursor.execute(sql)
        money_ratio = 1.0 - current_ratio
        if money_ratio > 0.00000099:
            sql       = 'replace into risk_asset_allocation_ratio (ra_alloc_id, ra_transfer_date, ra_fund_id, ra_fund_ratio,  created_at, updated_at) values (%d, "%s", %d, %f, "%s", "%s")' % (list_id, last_date, 30003446, money_ratio, datetime.now(), datetime.now())
           # cursor.execute(sql)

    #print all_code_position

    conn.commit()
    conn.close()

def getFee(fund_id,fee_type,amount,day=0):
    amount = float(amount)
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    if fee_type == 0:
        sql = 'select * from fund_fee where ff_code="%s" and ff_type=%s and ff_fee_type=1 order by ff_fee desc limit 1' % (fund_id.zfill(6),5)
        cur.execute(sql)
        result = cur.fetchone()
        fee = 0.003
        if result:
            fee = float(result['ff_fee'])
            if fee <= 0.003:
                return fee*amount
            else:
                fee = fee*0.2
                if fee>=0.003:
                    return fee*amount
                else:
                    return 0.003*amount
        else:
            sql = 'select * from fund_fee where ff_code="%s" and ff_type=%s and ff_fee_type=2 order by ff_fee asc limit 1' % (fund_id.zfill(6),5)
            cur.execute(sql)
            result = cur.fetchone()
            if result:
                fee = float(result['ff_fee'])
                return fee
        fund_type = getFundType(fund_id)
        if fund_type == 'huobi':
            return 0
        return fee*amount
    elif fee_type == 1:
        sql = 'select * from fund_fee where ff_code="%s" and ff_type=%s and ff_min_value<=%s and (ff_max_value>=%s or ff_max_value=null) order by ff_fee asc limit 1' % (fund_id.zfill(6),6,day,day)
        cur.execute(sql)
        result = cur.fetchone()
        if result:
            if result['ff_fee_type'] == 2:
                return float(result['ff_fee'])
            else:
                fee = float(result['ff_fee'])
                return fee*amount
        else:
            fund_type = getFundType(fund_id)
            if fund_type == 'huobi':
                return 0
            return 0.00*amount

def getCompany(fund_id):
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    sql = 'select * from fund_infos where fi_code="%s"' % fund_id.zfill(6)
    cur.execute(sql)
    result = cur.fetchone()
    if result:
        return result['fi_company_id']
    else:
        return 0

def isChangeOut(fund_id):
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    sql = 'select * from fund_infos where fi_code="%s"' % fund_id.zfill(6)
    cur.execute(sql)
    result = cur.fetchone()
    if result:
        if result['fi_yingmi_transfor_status'] in [0,2]:
            return True
    return False

def isChangeIn(fund_id):
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    sql = 'select * from fund_infos where fi_code="%s"' % fund_id.zfill(6)
    cur.execute(sql)
    result = cur.fetchone()
    if result:
        if result['fi_yingmi_transfor_status'] in [0,1]:
            return True
    return False

def getShare(fund_id,amount,day,count=0):
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    if count==0:
        sql = 'select * from wind_fund_value where wf_fund_code="%s" and wf_time>="%s" order by wf_time asc limit 1' % (fund_id.zfill(6),day)
    else:
        sql = 'select * from (select * from wind_fund_value where wf_fund_code="%s" and wf_time>="%s" order by wf_time asc limit %s ) as a order by wf_time desc limit 1' % (fund_id.zfill(6),day,count)
    cur.execute(sql)
    result = cur.fetchone()
    if result:
        return [result['wf_time'],amount/float(result['wf_nav_value'])]
    return [day,amount]

def getAmount(fund_id,share,day=0):
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    if day!=0:
        sql = 'select * from wind_fund_value where wf_fund_code="%s" and wf_time<="%s" order by wf_time desc limit 1' % (fund_id.zfill(6),day)
    else:
        sql = 'select * from wind_fund_value where wf_fund_code="%s" order by wf_time desc limit 1' % fund_id.zfill(6)
    cur.execute(sql)
    result = cur.fetchone()
    if result:
        return share * float(result['wf_nav_value'])
    return share

def getNavValue(fund_id,day):
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    sql = 'select * from wind_fund_value where wf_fund_code="%s" and wf_time<="%s" order by wf_time desc limit 1' % (fund_id.zfill(6),day)
    cur.execute(sql)
    result = cur.fetchone()
    if result:
        return float(result['wf_nav_value'])
    return 1

def getFundType(fund_id):
    type_list = {'zhishu':[2001010607,200101080102],'huobi':[20010104],'zhaiquan':[20010103,2001010203],'gupiao':[20010101,2001010201,2001010202,2001010204]}
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    sql = 'select * from wind_fund_type where wf_fund_code="%s" and wf_status=1' % fund_id.zfill(6)
    cur.execute(sql)
    result = cur.fetchone()
    if result:
        for k,v in list(type_list.items()):
            for i in v:
                if str(result['wf_type']).find(str(i)) >= 0:
                    return k
    print(('error---------------------------'+str(fund_id)))

def getBuyPoFee(fund_id):
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    sql = 'select * from fund_infos where fi_code="%s"' % fund_id.zfill(6)
    cur.execute(sql)
    result = cur.fetchone()
    if result:
        return float(result['fi_yingmi_amount'])
    return 1

def getRiskPosition(status):
    conn = MySQLdb.connect(**config.db_asset)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    if status==0:
        sql = 'select * from risk_asset_allocation_ratio order by ra_alloc_id,ra_transfer_date asc'
    else:
        sql = 'select * from risk_asset_allocation_ratio where ra_approve_status=5 order by ra_alloc_id,ra_transfer_date asc'
    cur.execute(sql)
    result = cur.fetchall()
    tmp = []
    if result:
        for i in result:
            tmp.append(tuple([i['ra_alloc_id'],str(i['ra_transfer_date'])+' 00:00:00',getFundCode(str(i['ra_fund_id'])),float(i['ra_fund_ratio'])]))
    return tmp

def getFundCode(fund_id):
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    sql = 'select * from fund_infos where fi_globalid="%s"' % fund_id
    cur.execute(sql)
    result = cur.fetchone()
    if result:
        return result['fi_code']

def insertNav(risk,position,risk_type):
    conn = MySQLdb.connect(**config.db_asset)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    if len(position)>0:
        if risk_type == 0:
            sql = 'delete from risk_asset_allocation_nav where ra_type=2 and ra_alloc_id=%s' % risk
            cur.execute(sql)
            for i in position:
                sql = 'insert into risk_asset_allocation_nav(ra_type,ra_alloc_id,ra_date,ra_nav,ra_inc) values(%s,%s,"%s",%s,%s)' % (2,risk,i['date'],round(i['nav'],6),round(i['inc'],4))
                cur.execute(sql)
        else:
            sql = 'delete from risk_asset_allocation_nav where ra_type=8 and ra_alloc_id=%s' % risk
            cur.execute(sql)
            for i in position:
                sql = 'insert into risk_asset_allocation_nav(ra_type,ra_alloc_id,ra_date,ra_nav,ra_inc) values(%s,%s,"%s",%s,%s)' % (8,risk,i['date'],round(i['nav'],6),round(i['inc'],4))
                cur.execute(sql)
    conn.commit()
    conn.close()

def getBuyStatus(days,funds):
    conn = MySQLdb.connect(**config.db_base)
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    days.append('2016-09-03')
    sql = 'select * from fund_status where fs_fund_code in (%s)  and fs_date in (%s) and  fs_subscribe_status in (0,6)' % (','.join(funds),'"'+'","'.join(days)+'"')
    cur.execute(sql)
    result = cur.fetchall()
    tmp = {}
    for row in  result:
        if str(row['fs_date']) in tmp:
            tmp[str(row['fs_date'])].append(row['fs_fund_code'])
        else:
            tmp[str(row['fs_date'])] = [row['fs_fund_code'],]
    return tmp
    



if __name__ == '__main__':


    '''
    allocationdata = AllocationData.allocationdata()
    df = pd.read_csv(datapath('stock_indicator_2015-07-10.csv'), index_col = 'code')
    allocationdata.stock_fund_measure['2015-07-10'] = df
    df = pd.read_csv(datapath('stock_label_2015-07-10.csv'), index_col = 'code')
    allocationdata.stock_fund_label['2015-07-10'] = df
    fund_measure(allocationdata)


    df = pd.read_csv(datapath('labelasset.csv'), index_col = 'date')
    allocationdata.label_asset_df = df
    df = pd.read_csv(datapath('stock_fund.csv'), index_col = 'date')
    allocationdata.stock_fund_df  = df
    df = pd.read_csv(datapath('equalriskasset.csv'), index_col = 'date')
    allocationdata.equal_risk_asset_df  = df
    df = pd.read_csv(datapath('equalriskassetratio.csv'), index_col = 'date')
    allocationdata.equal_risk_asset_ratio_df  = df
    df = pd.read_csv(datapath('bond_fund.csv'), index_col = 'date')
    allocationdata.bond_fund_df  = df


    label_asset(allocationdata)


    df = pd.read_csv(datapath('highriskasset.csv'), index_col = 'date')
    allocationdata.high_risk_asset_df = df
    df = pd.read_csv(datapath('lowriskasset.csv'), index_col = 'date')
    allocationdata.low_risk_asset_df = df
    df = pd.read_csv(datapath('highlowriskasset.csv'), index_col = 'date')
    allocationdata.highlow_risk_asset_df = df
    df = pd.read_csv(datapath('highriskposition.csv'), index_col = 'date')
    allocationdata.high_risk_position_df = df
    df = pd.read_csv(datapath('lowriskposition.csv'), index_col = 'date')
    allocationdata.low_risk_position_df = df
    df = pd.read_csv(datapath('highlowriskposition.csv'), index_col = 'date')
    allocationdata.highlow_risk_position_df = df


    asset_allocation(allocationdata)

    '''

    allocationdata = AllocationData.allocationdata()
    riskhighlowriskasset(allocationdata)
