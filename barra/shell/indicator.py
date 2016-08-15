#coding=utf8


import pandas as pd
import numpy as np
import string


fund_num = 30

def maxdrawdown(pvs):
    mdd = 0
    peak = pvs[0]
    for v in pvs:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > mdd:
            mdd = dd
    return mdd


if __name__ == '__main__':


    fund_df = pd.read_csv('./data/stock.csv', index_col='date', parse_dates=['date']).fillna(method='pad')
    fund_dfr = fund_df.pct_change().fillna(0.0)
    
    date_index = fund_dfr.index

    #dates = ['2014-12-31', '2015-02-06', '2015-03-13', '2015-04-17', '2015-05-22', '2015-06-26', '2015-07-31', '2015-09-02', '2015-10-09']

    dates = ['2014-12-31', '2015-02-06', '2015-03-13', '2015-04-17', '2015-05-22', '2015-06-26', '2015-07-31', '2015-09-02', '2015-10-09','2015-11-13', '2015-12-18', '2016-01-22', '2016-03-04', '2016-04-08', '2016-05-13', '2016-06-17', '2016-07-22']

    for j in range(0, len(date_index)):
        d = date_index[j]
        str_d = d.strftime('%Y-%m-%d')
        tmp_dfr = fund_dfr.iloc[ j - 52:j,] 
        tmp_df  = fund_df.iloc[ j - 52:j,] 
        if str_d in set(dates):
            name = './tmp/multi_factor_' + str_d + '.csv'
            df = pd.read_csv(name)
            df = df.sort(['score'],ascending = False)
            cs = df['code'].values
            data = []
            for i in range(0, fund_num):
                code = '%06d'% cs[i]
                rs = tmp_dfr[code]
                vs = tmp_df[code]
                mean = np.mean(rs) * 52
                std  = np.std(rs) * (52 ** 0.5)
                shape = (mean - 0.03) / std
                drawdown = maxdrawdown(vs)
                data.append([mean, std, shape, drawdown])
                 
            df = pd.DataFrame(data, index = cs[0:fund_num], columns = ['annual_return', 'std', 'shape','max_drawdown'])
            print d
            print df
            

 



    '''
    ranks = {}
    codes = set()
    for str_d in dates:
        #print str_d
        name = './tmp/multi_factor_' + str_d + '.csv'
        df = pd.read_csv(name)
        df = df.sort(['score'],ascending = False)
        codes = df['code'].values
        for i in range(0, 30):
            code = codes[i]
            print code
    '''
