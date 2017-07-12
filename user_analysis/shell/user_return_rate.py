#coding=utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('./tmp/online_nav.csv', index_col = ['date'], parse_dates = ['date'])

    day_interval = [180, 360, 720, 1080]


    for days in day_interval:

        gain = []
        loss = []
        rs = []
        max_drawdowns = []
        for i in range(0, len(df) - days):
            date = df.index[i]
            r = (df.iloc[i + days] / df.iloc[i] - 1).values
            rs.append(r)
            if r >= 0:
                gain.append(r)
            if r < 0:
                loss.append(r)

            tmp_df = df.iloc[i : i + days]
            tmp_df['max'] = tmp_df.cummax()
            tmp_df['drawdown'] = 1 - tmp_df['nav'] / tmp_df['max']
            max_drawdown = tmp_df['drawdown'].max()
            #print max_drawdown
            max_drawdowns.append(max_drawdown)

        print days, ',' ,np.mean(rs), ',', 1.0 * len(loss) / (len(loss) + len(gain)), ',', 1.0 * len(gain) / (len(loss) + len(gain)), ',', np.mean(loss), ',', np.mean(gain), ',', np.mean(max_drawdowns)
