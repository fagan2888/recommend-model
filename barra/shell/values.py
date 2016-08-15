#coding=utf8


import pandas as pd
import numpy as np
import string


fund_num = 30


if __name__ == '__main__':


    fund_df = pd.read_csv('./data/stock.csv', index_col='date', parse_dates=['date'])
    fund_dfr = fund_df.pct_change().fillna(0.0)
    
    date_index = fund_dfr.index

    #dates = ['2014-12-31', '2015-02-06', '2015-03-13', '2015-04-17', '2015-05-22', '2015-06-26', '2015-07-31', '2015-09-02', '2015-10-09']

    dates = ['2014-12-31', '2015-02-06', '2015-03-13', '2015-04-17', '2015-05-22', '2015-06-26', '2015-07-31', '2015-09-02', '2015-10-09','2015-11-13', '2015-12-18', '2016-01-22', '2016-03-04', '2016-04-08', '2016-05-13', '2016-06-17', '2016-07-22']
    codes = []

    vs = []
    pool_vs = []
    ds = []

    for d in date_index:
        str_d = d.strftime('%Y-%m-%d')
        if str_d in set(dates):
            name = './tmp/multi_factor_' + str_d + '.csv'
            df = pd.read_csv(name)
            df = df.sort(['score'],ascending = False)
            cs = df['code'].values
            codes = []
            for i in range(0, fund_num):
                code_str = '%06d' % cs[i]
                codes.append(code_str)

        r = 0.0           
        for code in codes:
            r = r + fund_dfr.loc[d, code] / len(codes)

        pool_r = 0.0           
        for code in fund_dfr.columns:
            pool_r = pool_r + fund_dfr.loc[d, code] / len(fund_dfr.columns)

        if len(vs) == 0:
            vs = [1]
            ds = [d]
            pool_vs = [1]
        else:
            v = vs[-1] * (1 + r)
            p_v = pool_vs[-1]  * ( 1 + pool_r)
            vs.append(v)
            pool_vs.append(p_v)
            ds.append(d)
            #print d, v


    df = pd.DataFrame(np.matrix([vs,pool_vs]).T, index = ds, columns = ['nav', 'p_nav'])
    df.to_csv('./tmp/values.csv')
 

   
        



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
