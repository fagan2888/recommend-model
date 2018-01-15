#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ipdb import set_trace


def re_view():
    #re_price = pd.read_csv('macro_data/re_area.csv', index_col = 0, parse_dates = True)
    re_price = pd.read_csv(
        'macro_data/re_price.csv',
        index_col=0,
        parse_dates=True)
    #re_sales = pd.read_csv('macro_data/re_sales.csv', index_col = 0, parse_dates = True)
    m1 = pd.read_csv('macro_data/m1.csv', index_col=0, parse_dates=True)
    re_price = re_price.diff(1).dropna()
    #re_sales = re_sales.diff(1).dropna()
    m1 = m1.rolling(6).mean().diff(1).dropna()

    #re_sales = re_sales.reindex(re_price.index)
    #re_sales = re_sales.fillna(method = 'pad')
    #re_sales = re_sales.fillna(0.0)

    m1 = m1.reindex(re_price.index)

    re_views = []
    for price, m1 in zip(re_price.iloc[:, 0], m1.iloc[:, 0]):
        # if price < 0:
        #    re_views.append(1)
        # else:
        #    re_views.append(-4)
        #tmp_price_view = 1 if price < 0 else (-2)
        #tmp_m1_view = 1 if m1 < 0 else (-2)
        if (price > 0) and (m1 > 0):
            re_views.append(-4)
        elif (price < 0) and (m1 < 0):
            re_views.append(2)
        elif (price > 0) and (m1 < 0):
            re_views.append(-2)
        else:
            re_views.append(0)

    re_price['re_view'] = re_views
    re_price = re_price.resample('m').fillna(method='pad')
    new_index = []
    for day in re_price.index:
        new_index.append(day + timedelta(15))
    re_price.index = new_index
    # re_price.to_csv('view/re_view.csv')
    return re_price.re_view


def ir_view():
    ir = pd.read_csv('macro_data/ir.csv', index_col=0, parse_dates=True)
    sf_m2 = pd.read_csv('macro_data/sf_m2.csv', index_col=0, parse_dates=True)

    ir = ir.diff(20).dropna()
    #sf_m2 = sf_m2.shift(3).diff(1).dropna()
    sf_m2 = sf_m2.rolling(12).mean().diff().dropna()
    new_index = []
    for day in sf_m2.index:
        new_index.append(day + timedelta(15))
    sf_m2.index = new_index

    ir_views = []
    dates = ir[ir.index >= '2012'].index
    ir = ir.loc[dates]
    for day in dates:
        tmp_ir = ir[ir.index <= day].values[-1, 0]
        tmp_sf_m2 = sf_m2[sf_m2.index <= day]
        if len(tmp_sf_m2) == 0:
            tmp_sf_m2 = 0
        else:
            tmp_sf_m2 = tmp_sf_m2.values[-1, 0]

        if (tmp_ir < 0) and (tmp_sf_m2 < 0):
            ir_views.append(2)
        elif (tmp_ir > 0) and (tmp_sf_m2 > 0):
            ir_views.append(-1)
        else:
            ir_views.append(0)

        # if tmp_sf_m2 < 0:
        #    ir_views.append(2)
        # elif tmp_sf_m2 > 0:
        #    ir_views.append(-2)
        # else:
        #    ir_views.append(0)

    ir['ir_view'] = ir_views
    return ir.ir_view


def eps_view():
    eps = pd.read_csv('macro_data/eps.csv', index_col=0, parse_dates=True)
    eps = reindex_eps(eps)
    eps = eps.resample('m').fillna(method='pad')
    #eps_views = []
    #dates = eps.index
    #eps['eps_view'] = np.sign(eps.eps)

    return eps


def pe_view():
    pe = pd.read_csv('macro_data/pe.csv', index_col=0, parse_dates=True)
    bfid = pd.read_csv('macro_data/bfid.csv', index_col=0, parse_dates=True)

    new_index = bfid.index + timedelta(15)
    bfid.index = new_index
    bfid = bfid.resample('m').last()

    pe = pe.resample('m').last()
    pe_bfid = pd.merge(bfid, pe, left_index=True, right_index=True, how='left')
    pe_bfid['pe_ttm_inv'] *= 100
    pe_bfid['erp'] = pe_bfid['pe_ttm_inv'] - pe_bfid['bfid']

    eps = eps_view()
    #eps['eps_chg'] = eps.eps.rolling(4).apply(lambda x:(x[-1]-x[0])*100/x[0])*4
    #eps['eps_chg'] = eps.eps.rolling(13).apply(lambda x:(x[-1]-x[0])*100/x[0])
    pe_bfid_eps = pd.merge(
        pe_bfid,
        eps,
        left_index=True,
        right_index=True,
        how='inner')
    pe_bfid_eps = pe_bfid_eps.dropna()

    pe_bfid_eps['view'] = pe_bfid_eps['erp'] + pe_bfid_eps['eps_chg']
    pe_bfid_eps['view'] = np.sign(pe_bfid_eps['view']) * 3

    return pe_bfid_eps.view


def reindex_eps(eps):
    eps = eps.resample('3m').last()
    dates = eps.index
    redates = []
    for date in dates:
        if date.month == 12:
            tmp_date = date + timedelta(90)
        elif date.month == 3:
            tmp_date = date + timedelta(30)
        elif date.month == 6:
            tmp_date = date + timedelta(60)
        elif date.month == 9:
            tmp_date = date + timedelta(30)

        redates.append(tmp_date)
    eps.index = redates
    eps['eps_chg'] = eps.pct_change() * 100 * 4
    #eps = eps.resample('m').fillna(method = 'pad')
    return eps


def eri_view():
    eri = pd.read_csv('macro_data/eri.csv', index_col=0, parse_dates=True)
    eri = eri.diff(20)
    eri_view = eri.eri.apply(lambda x: -1 if x > 0 else 0)

    # eri_view.to_csv("tmp/tmp_eri.csv")
    # print eri_view
    return eri_view


def macro_view():
    rev = re_view()
    irv = ir_view()
    pev = pe_view()
    eriv = eri_view()
    re_views = []
    ir_views = []
    pe_views = []
    eri_views = []
    macro_views = []
    dates = irv.index
    for day in dates:
        tmp_rev = rev[rev.index <= day].values[-1]
        tmp_irv = irv[irv.index <= day].values[-1]
        tmp_pev = pev[pev.index <= day].values[-1]
        tmp_eriv = eriv[eriv.index <= day].values[-1]

        re_views.append(tmp_rev)
        ir_views.append(tmp_irv)
        pe_views.append(tmp_pev)
        eri_views.append(tmp_eriv)
        macro_views.append(tmp_rev + tmp_irv + tmp_pev + tmp_eriv)

    result_df = pd.DataFrame(
        data=np.column_stack([
            macro_views,
            re_views,
            ir_views,
            pe_views,
            eri_views
        ]),

        columns=[
            'macro_view',
            're_view',
            'ir_view',
            'eps_view',
            'eri_view'
        ],
        index=dates
    )
    result_df.to_csv('view/macro_view.csv', index_label='date')

    return result_df


def rotation_view():
    rot = pd.read_csv('macro_data/rot.csv', index_col=0, parse_dates=True)
    rot['sz50_nav'] = rot.sz50 / rot.sz50[0]
    rot['zxb_nav'] = rot.zxb / rot.zxb[0]
    rot['nav_diff'] = rot.sz50_nav - rot.zxb_nav
    rot['nav_diff_mean_120'] = rot['nav_diff'].rolling(120).mean()
    rot['rot'] = rot['nav_diff_mean_120'].diff(5)
    rot['rot_view'] = np.sign(rot['rot'])
    rot = rot.dropna()

    rot[['rot_view']].to_csv('view/rot_view.csv', index_label='date')

    return rot.rot_view


if __name__ == '__main__':
    #rev = re_view('macro_data/re_price.csv')
    # print rev['2015']
    # ir_view()

    # eps_view()
    # pe_view()
    # eri_view()

    macro_view()
    # rotation_view()
