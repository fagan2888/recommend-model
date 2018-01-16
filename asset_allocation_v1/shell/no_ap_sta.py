#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
from db import base_ra_index_nav, asset_on_online_nav, asset_on_online_fund, base_ra_fund_nav
from datetime import datetime, timedelta
from ipdb import set_trace

def cal_diff():
    pf = asset_on_online_nav.load_series('800000', 8, begin_date = '2017-01-01')
    pf = pd.DataFrame(pf, index = pf.index)

    fp = asset_on_online_fund.load_fund_pos('800000')
    fp = fp.reset_index()
    fp = fp.set_index('on_date')
    pos = fp['2016-12']
    fund_ids = pos.on_fund_id.values
    fund_poses = pos.on_fund_ratio.values

    nav = None
    for id_, pos in zip(fund_ids, fund_poses):
        nav_fund= base_ra_fund_nav.load_daily('2016-12-30', '2017-12-29', [id_])
        if  nav is None:
            nav = nav_fund.values.ravel()*pos
        else:
            nav += nav_fund.values.ravel()*pos

    set_trace()



    return pf


if __name__ == '__main__':
    cal_diff()
