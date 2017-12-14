#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ipdb import set_trace


def re_view():
    #re_sales = pd.read_csv('macro_data/re_area.csv', index_col = 0, parse_dates = True)
    re_sales = pd.read_csv('macro_data/re_price.csv', index_col = 0, parse_dates = True)
    re_sales = re_sales.diff(1).dropna()
    re_views = []
    for sale in re_sales.iloc[:, 0]:
        if sale < 0:
            re_views.append(1)
        else:
            re_views.append(-4)
    re_sales['re_view'] = re_views
    re_sales = re_sales.resample('m').fillna(method = 'pad')
    new_index = []
    for day in re_sales.index:
        new_index.append(day+timedelta(15))
    re_sales.index = new_index
    #re_sales.to_csv('view/re_view.csv')
    return re_sales


def ir_view():
    ir = pd.read_csv('macro_data/ir.csv', index_col = 0, parse_dates = True)
    sf_m2 = pd.read_csv('macro_data/sf_m2.csv', index_col = 0, parse_dates = True)

    ir = ir.diff(20).dropna()
    sf_m2 = sf_m2.shift(3).diff(1).dropna()
    new_index = []
    for day in sf_m2.index:
        new_index.append(day+timedelta(15))
    sf_m2.index = new_index
    
    ir_views = []
    dates = ir[ir.index >= '2012'].index
    ir = ir.loc[dates]
    for day in dates:
        tmp_ir = ir[ir.index <= day].values[-1, 0]
        tmp_sf_m2 = sf_m2[sf_m2.index <= day].values[-1, 0]
        if (tmp_ir < 0) and (tmp_sf_m2 < 0):
            ir_views.append(2)
        elif (tmp_ir > 0) and (tmp_sf_m2 > 0):
            ir_views.append(-2)
        else:
            ir_views.append(0)

        #if tmp_sf_m2 < 0:
        #    ir_views.append(2)
        #elif tmp_sf_m2 > 0:
        #    ir_views.append(-2)
        #else:
        #    ir_views.append(0)

    ir['ir_view'] = ir_views
    return ir


def macro_view():
    rev = re_view()
    irv = ir_view()
    dates = irv.index
    macro_views = []
    for day in dates:
        tmp_rev = rev[rev.index <= day].values[-1, -1]
        tmp_irv = irv[irv.index <= day].values[-1, -1]
        macro_views.append(tmp_rev+tmp_irv)
    result_df = pd.DataFrame(data = macro_views, columns = ['macro_view'], index = dates)     
    result_df.to_csv('view/macro_view.csv', index_label = 'date')

    return result_df

if __name__ == '__main__':
    #rev = re_view('macro_data/re_price.csv')
    #print rev['2015']
    #ir_view()
    macro_view()
