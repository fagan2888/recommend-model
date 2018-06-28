#coding=utf8


import sys
sys.path.append("shell")
import Const
import string
from numpy import *
import numpy as np
import pandas as pd
import Financial as fin
import FundIndicator as fi
from Const import datapath

fund_num = Const.fund_num

def select_stock_new(day, df_label, df_indicator, limit=5):
    daystr = day.strftime("%Y-%m-%d")
    # df = df_label.merge(df_indicator, left_index=True, right_index=True)
    categories = ['largecap','smallcap','rise','decline','oscillation','growth','value']
    
    data = {}
    for category in categories:
        index_codes = df_label[df_label[category] == 1].index
        df_tmp = df_indicator.loc[index_codes]
        data[category] = df_tmp.sort_values(by='jensen', ascending=False)[0:limit]
        
    df_result = pd.concat(data, names=['category','code'])
    #df_result.to_csv(datapath('stock_pool_' + daystr + '.csv'))
    
    return df_result


def select_bond_new(day, df_label, df_indicator, limit=5):
    daystr = day.strftime("%Y-%m-%d")
    # df = df_label.merge(df_indicator, left_index=True, right_index=True)
    categories = ['ratebond','creditbond','convertiblebond']
    
    data = {}
    for category in categories:
        index_codes = df_label[df_label[category] == 1].index
        df_tmp = df_indicator.loc[index_codes]
        data[category] = df_tmp.sort_values(by='jensen', ascending=False)[0:limit]
        
    df_result = pd.concat(data, names=['category','code'])
    df_result.to_csv(datapath('bond_pool_' + daystr + '.csv'))
    
    return df_result

def select_money_new(day, df_indicator, limit=1):
    daystr = day.strftime("%Y-%m-%d")
    # df = df_label.merge(df_indicator, left_index=True, right_index=True)
    categories = ['money']
    
    data = {}
    for category in categories:
        data[category] = df_indicator.sort_values(by='sharpe', ascending=False)[:limit]
        
    df_result = pd.concat(data, names=['category','code'])
    df_result.to_csv(datapath('money_pool_' + daystr + '.csv'))
    
    return df_result

def select_other_new(day, df_indicator):
    daystr = day.strftime("%Y-%m-%d")
    categories = ['SP500.SPI','GLNC','HSCI.HI']
    
    data = {}
    for category in categories:
        data[category] = df_indicator.loc[[category]]
        
    df_result = pd.concat(data, names=['category','code'])
    df_result.to_csv(datapath('other_pool_' + daystr + '.csv'))
    
    return df_result
