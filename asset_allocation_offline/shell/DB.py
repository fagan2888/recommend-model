#coding=utf8


import sys
sys.path.append('shell')
import MySQLdb
import string
import pandas as pd
import numpy as np
from datetime import datetime
import FundIndicator
import AllocationData
import os



def fund_measure(allocationdata):


    conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
    cursor = conn.cursor()


    base_sql = 'replace into fund_measure (fm_start_date, fm_end_date, fm_adjust_period, fm_look_back, fm_fund_type, fm_fund_code, fm_jensen, fm_ppw, fm_stability, fm_sortino, fm_sharpe, fm_high_postion_prefer, fm_low_position_prefer, fm_largecap_prefer, fm_smallcap_prefer, fm_growth_prefer, fm_value_prefer, fm_largecap_fitness, fm_smallcap_fitness, fm_rise_fitness, fm_decline_fitness, fm_oscillation_fitness, fm_growth_fitness, fm_value_fitness, fm_ratebond, fm_creditbond, fm_convertiblebond, fm_sp500, fm_gold, fm_hs, created_at, updated_at) values ("%s","%s",%d, %d, "%s","%s", %f,%f,%f,%f,%f, %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d, "%s","%s")'


    stock_fund_measure = allocationdata.stock_fund_measure
    stock_fund_label   = allocationdata.stock_fund_label
    bond_fund_measure  = allocationdata.bond_fund_measure
    bond_fund_label    = allocationdata.bond_fund_label


    dates = list(stock_fund_measure.keys())
    dates.sort()

    
    for date in dates:

        stock_measure_df = stock_fund_measure[date]    
        stock_label_df   = stock_fund_label[date]    

        for code in stock_fund_measure[date].index:

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

    conn.commit()
    conn.close()

    print 'fund measure done'



def label_asset(allocationdata):


    conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
    cursor = conn.cursor()


    label_asset_df            = allocationdata.label_asset_df
    stock_fund_df             = allocationdata.stock_fund_df
    bond_fund_df              = allocationdata.bond_fund_df
    equal_risk_asset_ratio_df = allocationdata.equal_risk_asset_ratio_df
    equal_risk_asset_df       = allocationdata.equal_risk_asset_df


    dates = equal_risk_asset_df.index.values
    dates.sort()
    d = dates[0]

    
    label_asset_df = label_asset_df[label_asset_df.index > d]
    dates = label_asset_df.index.values
    dates.sort()


    columns = label_asset_df.columns
    values = []
    for col in columns:
        vs = [1]    
        for d in dates:
            r = label_asset_df.loc[d, col]
            v = vs[-1] * ( 1 + r)
            vs.append(v)    
        values.append(vs)    


    m = np.matrix(values)
    label_asset_df = pd.DataFrame(m.T, index = equal_risk_asset_df.index.values, columns = columns)

        
    base_sql = 'replace into fixed_risk_asset (fra_start_date, fra_adjust_period, fra_asset_look_back, fra_risk_adjust_period, fra_risk_look_back, fra_fund_type, fra_fund_code, fra_asset_type, fra_position, fra_asset_label, fra_net_value, fra_date, fra_annual_return, fra_sharpe, fra_std, fra_maxdrawdown ,created_at, updated_at) values ("%s", %d, %d, %d, %d, "%s","%s", "%s" ,%f, "%s", %f, "%s", %f, %f, %f, %f, "%s", "%s")'


    stock_tags        = ['largecap','smallcap','rise','oscillation','decline','growth','value']
    #origin_bond_tags  = ['ratebond','creditbond','convertiblebond']    
    origin_bond_tags  = ['ratebond','creditbond']    
    #bond_tags         = ['convertiblebond']
    money_tags        = ['money']
    other_tags        = ['SP500.SPI','SPGSGCTR.SPI','HSCI.HI']


    stock_fund_df_dates = set(stock_fund_df.index.values)
    equal_risk_asset_ratio_dates = set(equal_risk_asset_ratio_df.index.values)    


    for label in stock_tags:

        fund = ''
        dates = label_asset_df.index.values
        dates.sort()

        navs = []

        for d in dates:
            if d in stock_fund_df_dates:
                fund = stock_fund_df.loc[d, label]

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
            sql = base_sql % (allocationdata.start_date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback, allocationdata.fixed_risk_asset_risk_adjust_period, allocationdata.fixed_risk_asset_risk_lookback, 'stock', fund, 'origin', 1.0, label, net_value, d, returns, sharpe, risk, maxdrawdown , datetime.now(), datetime.now())

            cursor.execute(sql)


    for label in stock_tags:

        fund = ''
        position = 0.0
        dates = label_asset_df.index.values
        dates.sort()

        navs = []

        for d in dates:

            if d in stock_fund_df_dates:
                fund = stock_fund_df.loc[d, label]
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
            sql = base_sql % (allocationdata.start_date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback, allocationdata.fixed_risk_asset_risk_adjust_period, allocationdata.fixed_risk_asset_risk_lookback, 'stock', fund, 'fixed_risk', position, label, net_value, d, returns, sharpe, risk, maxdrawdown, datetime.now(), datetime.now())

            #print sql
            cursor.execute(sql)


    bond_fund_df_dates = set(bond_fund_df.index.values)

    for label in origin_bond_tags:

        fund = ''
        dates = label_asset_df.index.values
        dates.sort()

        navs = []

        for d in dates:

            if d in bond_fund_df_dates:
                fund = bond_fund_df.loc[d, label]

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
            sql = base_sql % (allocationdata.start_date, allocationdata.fund_measure_adjust_period, allocationdata.fund_measure_lookback, allocationdata.fixed_risk_asset_risk_adjust_period, allocationdata.fixed_risk_asset_risk_lookback, 'bond', fund, 'origin', 1.0, label, net_value, d, returns, sharpe, risk, maxdrawdown, datetime.now(), datetime.now())    
            
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


    print 'label asset done'

def get_fund_value(fund_ids):
    conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='mofang', charset='utf8')
        cursor = conn.cursor()
    base_sql = "select wf_fund_id, wf_fund_code,  wf_time, wf_nav_value from wind_fund_value where wf_fund_id in %s"
    sql = base_sql % fund_ids
    ret = cursor.execute(base_sql, fund_ids)
    ret = cursor.fetchall()
    conn.commit()
    conn.close()
    print ret
def asset_allocation(allocationdata):

    
    conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
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
    credit_bond_ratio       = 0.0
    SP500_SPI_ratio        = 0.0
    SPGSGCTR_SPI_ratio     = 0.0
    HSCI_HI_ratio          = 0.0    
    highrisk_ratio         = 0.0
    lowrisk_ratio          = 0.0


    navs = []

    dates = high_risk_asset_df.index.values
    dates.sort()
    high_position_dates = set(high_risk_position_df.index.values)

    #print high_risk_position_df

    for d in dates:
        #print high_risk_asset_df
        net_value = high_risk_asset_df.loc[d, 'high_risk_asset']
        navs.append(net_value)
        if d in high_position_dates:
            largecap_ratio = high_risk_position_df.loc[d, 'largecap']
            smallcap_ratio = high_risk_position_df.loc[d, 'smallcap']
            rise_ratio = high_risk_position_df.loc[d, 'rise']
            oscillation_ratio = high_risk_position_df.loc[d, 'oscillation']
            decline_ratio = high_risk_position_df.loc[d, 'decline']
            growth_ratio = high_risk_position_df.loc[d, 'growth']
            value_ratio = high_risk_position_df.loc[d, 'value']
            #convertible_ratio = high_risk_position_df.loc[d, 'convertiblebond']
            SP500_SPI_ratio = high_risk_position_df.loc[d, 'SP500.SPI']
            SPGSGCTR_SPI_ratio = high_risk_position_df.loc[d, 'GLNC']
            HSCI_HI_ratio = high_risk_position_df.loc[d, 'HSCI.HI']


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
        if d in low_position_dates:
            rate_bond_ratio   = low_risk_position_df.loc[d, 'ratebond']
            credit_bond_ratio = low_risk_position_df.loc[d, 'creditbond']


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
        if d in low_position_dates:
            highrisk_ratio   =  highlow_risk_position_df.loc[d, 'high_risk_asset']
            lowrisk_ratio    =  highlow_risk_position_df.loc[d, 'low_risk_asset']


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


        #print sql
        cursor.execute(sql)


    conn.commit()
    conn.close()

    print 'asset allocation done'

    return 0    




if __name__ == '__main__':



    
    allocationdata = AllocationData.allocationdata()
    df = pd.read_csv('./tmp/stock_indicator_2015-07-10.csv', index_col = 'code')
    allocationdata.stock_fund_measure['2015-07-10'] = df
    df = pd.read_csv('./tmp/stock_label_2015-07-10.csv', index_col = 'code')
    allocationdata.stock_fund_label['2015-07-10'] = df
    fund_measure(allocationdata)




    df = pd.read_csv('./tmp/labelasset.csv', index_col = 'date')
    allocationdata.label_asset_df = df
    df = pd.read_csv('./tmp/stock_fund.csv', index_col = 'date')
    allocationdata.stock_fund_df  = df
    df = pd.read_csv('./tmp/equalriskasset.csv', index_col = 'date')
    allocationdata.equal_risk_asset_df  = df
    df = pd.read_csv('./tmp/equalriskassetratio.csv', index_col = 'date')
    allocationdata.equal_risk_asset_ratio_df  = df
    df = pd.read_csv('./tmp/bond_fund.csv', index_col = 'date')
    allocationdata.bond_fund_df  = df


    label_asset(allocationdata)




    df = pd.read_csv('./tmp/highriskasset.csv', index_col = 'date')
    allocationdata.high_risk_asset_df = df
    df = pd.read_csv('./tmp/lowriskasset.csv', index_col = 'date')
    allocationdata.low_risk_asset_df = df
    df = pd.read_csv('./tmp/highlowriskasset.csv', index_col = 'date')
    allocationdata.highlow_risk_asset_df = df
    df = pd.read_csv('./tmp/highriskposition.csv', index_col = 'date')
    allocationdata.high_risk_position_df = df
    df = pd.read_csv('./tmp/lowriskposition.csv', index_col = 'date')
    allocationdata.low_risk_position_df = df
    df = pd.read_csv('./tmp/highlowriskposition.csv', index_col = 'date')
    allocationdata.highlow_risk_position_df = df


    asset_allocation(allocationdata)
