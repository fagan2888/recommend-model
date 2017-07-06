from __future__ import division
import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings('ignore')

def get_signal_wr(df):
    win_num = 0
    total_num = 0
    signal_nav = 1

    for i in range(len(df)-1):
        current = df.iloc[:, -1][i]
        next_ = df.iloc[:, -1][i+1]

        if current == next_:
            signal_nav *= (1 + df['pct_chg'][i+1]/100)
        elif current != next_:
            signal_nav *= (1 + df['pct_chg'][i+1]/100)
            if (signal_nav-1)*current >= 0:
                win_num += 1
            total_num += 1
            signal_nav = 1

    #win_ratio = win_num/total_num
    return win_num, total_num


def cal_state_wr(df, market_div):
    '''
    win_down = 0,0
    total_down = 0,1
    win_osc = 1,0
    total_osc = 1,1
    win_up = 2,0
    total_up = 2,1
    '''

    result = np.zeros(6).reshape(3,2)
    for time, state in market_div:
        #print df.loc[time, :]
        win, total = get_signal_wr(df.loc[time, :])
        result[state,0] += win
        result[state,1] += total

    win, total = get_signal_wr(df)

    print 'wr: ', win/total
    print 'wr_down: ', result[0,0]/result[0,1]
    print 'wr_osc: ', result[1,0]/result[1,1]
    print 'wr_up: ', result[2,0]/result[2,1]


if __name__ == '__main__':

    assets = ['120000001', '120000002','120000013','120000014','120000015',\
            '120000029']

    market_div_sh300 = \
            [[slice('20010101','20140711'), 1], [slice('20140718','20150605'),2], \
            [slice('20150612','20160626'), 0], [slice('20160304','20171231'),1]]

    market_div_zz500 = \
            [[slice('20010101','20140620'), 1], [slice('20140627','20150612'),2], \
            [slice('20150619','20160918'), 0], [slice('20160925','20161225'),2], \
            [slice('20161231','20170129'),0], [slice('20170205','20171231'),1]]

    market_div_sp500 = \
            [[slice('20010101','20150807'), 2], [slice('20150814','20161104'),1], \
            [slice('20161111','20171231'), 2]]

    market_div_au = \
            [[slice('20010101','20130628'), 0], [slice('20130705','20151218'),1], \
            [slice('20151225','20171231'), 2]]

    market_div_hsci = \
            [[slice('20010101','20130621'), 1], [slice('20130628','20150430'),2], \
            [slice('20150508','20160122'), 0], [slice('20160129','20171231'), 2]]

    market_div_nhsp = \
            [[slice('20010101','20151120'), 0], [slice('20151127','20171231'),2]]

    market_div = [market_div_sh300, market_div_zz500, market_div_sp500, \
            market_div_au, market_div_hsci, market_div_nhsp]

    for asset, market_state in zip(assets, market_div):
        df = pd.read_csv('./output_data/' + asset + '_hmm.csv', \
                index_col = 0, parse_dates = True)

        print 'asset: ', asset
        cal_state_wr(df, market_state)
        print
