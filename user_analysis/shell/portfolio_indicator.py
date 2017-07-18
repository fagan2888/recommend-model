#coding=utf8


import pandas as pd
import numpy as np


def max_drawdown(tmp_df):

    tmp_df['max'] = tmp_df.cummax()
    tmp_df['drawdown'] = 1 - tmp_df['nav'] / tmp_df['max']
    max_dn = tmp_df['drawdown'].max()
    return max_dn


def ret(tmp_df):
    return (tmp_df.iloc[-1] / tmp_df.iloc[0]).values[0] - 1


def nav_sharpe(tmp_df):
    tmp_dfr = tmp_df['nav'].pct_change().fillna(0.0)
    sharpe = ((np.mean(tmp_dfr) - 0.03 / 360) /  np.std(tmp_dfr)) * (360 ** 0.5)
    return sharpe


if __name__ == '__main__':

    df = pd.read_csv('./tmp/portfolio_nav.csv', index_col = ['date'], parse_dates = ['date'])

    day_interval = [30, 90, 180, 360, 720, 1800]

    for days in day_interval:

        rs = []
        max_drawdowns = []
        r_maxdns = []
        sharpes = []

        for i in range(0, len(df) - days - 1):

            date = df.index[i]
            tmp_df = df.iloc[i : i + days + 1]

            r = ret(tmp_df)
            maxdn = max_drawdown(tmp_df)
            r_maxdn = (r / days) * 360 / maxdn
            sharpe = nav_sharpe(tmp_df)

            rs.append(r)
            max_drawdowns.append(maxdn)
            r_maxdns.append(r_maxdn)
            sharpes.append(sharpe)

        rs = np.array(rs)
        print days, ',' ,1.0 * sum(rs > 0) / len(rs), ',', np.mean(max_drawdowns), ',', np.mean(sharpes) ,',', np.mean(r_maxdns)
