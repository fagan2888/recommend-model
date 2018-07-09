#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import pandas as pd
import numpy as np
from ipdb import set_trace
from db import asset_mz_markowitz_nav


def load_factor_index():
    mz_ids = ['MZ.SF40%d0'%i for i in range(1, 10)]+['MZ.SF50%d0'%i for i in range(1, 10)]
    recolumns = ['SF.00000%d.1'%i for i in range(1, 10)]+['SF.00000%d.0'%i for i in range(1, 10)]
    df_res = {}
    for mz_id in mz_ids:
        df_res[mz_id] = asset_mz_markowitz_nav.load_series(mz_id)
    df_res = pd.DataFrame(df_res)
    df_res.columns = recolumns

    return df_res


if __name__ == '__main__':

    load_factor_index()
