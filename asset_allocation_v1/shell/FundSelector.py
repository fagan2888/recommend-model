#coding=utf8


import sys
sys.path.append("windshell")
import main
import Data
import Const
import string
from numpy import *
import numpy as np
import pandas as pd
import Financial as fin
import FundIndicator as fi
from Const import datapath

fund_num = Const.fund_num

def select_stock_new(day, df_label, df_indicator):
    daystr = day.strftime("%Y-%m-%d")
    # df = df_label.merge(df_indicator, left_index=True, right_index=True)
    categories = ['largecap','smallcap','rise','decline','oscillation','growth','value']
    
    data = {}
    for category in categories:
        index_codes = df_label[df_label[category] == 1].index
        #print index_codes
        #print df_indicator.index
        df_tmp = df_indicator.loc[index_codes]
        data[category] = df_tmp.sort_values(by='jensen', ascending=False)[0:fund_num]
        
    df_result = pd.concat(data, names=['category','code'])
    df_result.to_csv(datapath('stock_pool_' + daystr + '.csv'))
    
    return df_result

def select_stock(funddf, fund_tags, indexdf):

    largecap_codes             = fund_tags['largecap']
    smallcap_codes             = fund_tags['smallcap']
    risefitness_codes          = fund_tags['risefitness']
    declinefitness_codes       = fund_tags['declinefitness']
    oscillationfitness_codes   = fund_tags['oscillationfitness']
    growthfitness_codes        = fund_tags['growthfitness']
    valuefitness_codes         = fund_tags['valuefitness']


    need_largecap           = True
    need_smallcap           = True
    need_risefitness        = True
    need_declinefitness     = True
    need_oscillationfitness = True
    need_growthfitness      = True
    need_valuefitness       = True

    #fund_sharpe  = fi.fund_sharp_annual(funddf)
    fund_sharpe  = fi.fund_jensen(funddf, indexdf)

    codes = []
    tag   = {}

    names = ['largecap','smallcap','rise','decline','oscillation','growth','value']
    for name in names:
        tag.setdefault(name, [])

    for i in range(0, len(fund_sharpe)):

        code = fund_sharpe[i][0]
        if code in set(largecap_codes) and len(tag['largecap']) < fund_num:
            codes.append(code)
            need_largecap = False
            tmp = tag.setdefault('largecap', [])
            tmp.append(code)
            #tag['largecap'] = code
            #print code, i
            #continue
        if code in set(smallcap_codes) and len(tag['smallcap']) < fund_num:
            codes.append(code)
            need_smallcap = False
            tmp = tag.setdefault('smallcap', [])
            tmp.append(code)
            #print code, i
            #continue
        if code in set(risefitness_codes) and len(tag['rise']) < fund_num:
            codes.append(code)
            need_risefitness = False
            tmp = tag.setdefault('rise', [])
            tmp.append(code)
            #print code, i
            #continue
        if code in set(declinefitness_codes) and len(tag['decline']) < fund_num:
            codes.append(code)
            need_declinefitness = False
            tmp = tag.setdefault('decline', [])
            tmp.append(code)
            #print code, i
            #continue
        if code in set(oscillationfitness_codes) and len(tag['oscillation']) < fund_num:
            codes.append(code)
            need_oscillationfitness = False
            tmp = tag.setdefault('oscillation', [])
            tmp.append(code)
            #print code, i
            #continue
        if code in set(growthfitness_codes) and len(tag['growth']) < fund_num:
            codes.append(code)
            need_growthfitness = False
            tmp = tag.setdefault('growth', [])
            tmp.append(code)
            #print code, i
            #continue
        if code in set(valuefitness_codes) and len(tag['value']) < fund_num:
            codes.append(code)
            need_valuefitness = False
            tmp = tag.setdefault('value', [])
            tmp.append(code)
            #print code, i
            #continue

    #print fund_sharpe
    #print codes

    return codes, tag


def select_bond_new(day, df_label, df_indicator):
    daystr = day.strftime("%Y-%m-%d")
    # df = df_label.merge(df_indicator, left_index=True, right_index=True)
    categories = ['ratebond','creditbond','convertiblebond']
    
    data = {}
    for category in categories:
        index_codes = df_label[df_label[category] == 1].index
        df_tmp = df_indicator.loc[index_codes]
        data[category] = df_tmp.sort_values(by='jensen', ascending=False)[0:fund_num]
        
    df_result = pd.concat(data, names=['category','code'])
    df_result.to_csv(datapath('bond_pool_' + daystr + '.csv'))
    
    return df_result

def select_bond(funddf, fund_tags, indexdf):

    ratebond_codes             = fund_tags['ratebond']
    credit_codes               = fund_tags['creditbond']
    convertible_codes          = fund_tags['convertiblebond']

    #fund_sharpe  = fi.fund_sharp_annual(funddf)
    fund_sharpe  = fi.fund_jensen(funddf, indexdf)
    codes = []
    tag   = {}
    for i in range(0, len(fund_sharpe)):
        code = fund_sharpe[i][0]

    codes = []
    tag   = {}

    names = ['ratebond','creditbond','convertiblebond']
    for name in names:
        tag.setdefault(name, [])

    for i in range(0, len(fund_sharpe)):

        code = fund_sharpe[i][0]
        #print code
        if code in set(ratebond_codes) and len(tag['ratebond']) < fund_num:
            codes.append(code)
            need_rate = False
            tmp = tag.setdefault('ratebond', [])
            tmp.append(code)
            #print code, i
            #continue
        if code in set(credit_codes) and len(tag['creditbond']) < fund_num:
            codes.append(code)
            need_credit = False
            tmp = tag.setdefault('creditbond', [])
            tmp.append(code)
            #print code, i
            #continue
        if code in set(convertible_codes) and len(tag['convertiblebond']) < fund_num:
            codes.append(code)
            need_convertible = False
            tmp = tag.setdefault('convertiblebond', [])
            tmp.append(code)
            #print code, i
            #continue

    #print fund_sharpe
    #print codes

    return codes, tag

def select_money_new(day, df_indicator):
    daystr = day.strftime("%Y-%m-%d")
    # df = df_label.merge(df_indicator, left_index=True, right_index=True)
    categories = ['money']
    
    data = {}
    for category in categories:
        data[category] = df_indicator.sort_values(by='sharpe', ascending=False)[:1]
        
    df_result = pd.concat(data, names=['category','code'])
    df_result.to_csv(datapath('money_pool_' + daystr + '.csv'))
    
    return df_result

def select_money(funddf):

    day = funddf.index.max()
    daystr = day.strftime("%Y-%m-%d")
    
    fund_sharpe  = fi.fund_sharp_annual(funddf)
    print fund_sharpe
    df = pd.DataFrame(fund_sharpe, columns=['code', 'sharpe'])
    df.set_index('code', inplace=True)
    df.to_csv(datapath('money_indicator_' + daystr + '.csv'))
    
    codes = []
    tag   = {}
    tag['money'] = fund_sharpe[0][0]
    codes.append(fund_sharpe[0][0])
    #tag['sharpe2'] = fund_sharpe[1][0]
    #codes.append(fund_sharpe[0][0])

    return codes, tag

def select_other_new(day, df_indicator):
    daystr = day.strftime("%Y-%m-%d")
    categories = ['SP500.SPI','GLNC','HSCI.HI']
    
    data = {}
    for category in categories:
        data[category] = df_indicator.loc[[category]]
        
    df_result = pd.concat(data, names=['category','code'])
    df_result.to_csv(datapath('other_pool_' + daystr + '.csv'))
    
    return df_result
