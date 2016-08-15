#coding=utf8


import pandas as pd
import numpy as np

fund_num = 30 

if __name__ == '__main__':


    dates = ['2014-12-31', '2015-02-06', '2015-03-13', '2015-04-17', '2015-05-22', '2015-06-26', '2015-07-31', '2015-09-02', '2015-10-09','2015-11-13', '2015-12-18', '2016-01-22', '2016-03-04', '2016-04-08', '2016-05-13', '2016-06-17', '2016-07-22']


    ranks = {}
    pre_codes = set()
    for str_d in dates:
        #print str_d
        name = './tmp/multi_factor_' + str_d + '.csv'
        df = pd.read_csv(name)
        df = df.sort(['score'],ascending = False)
        codes = df['code'].values
        codes = codes[0:fund_num]

        '''
        print str_d, len(pre_codes & set(codes))
        pre_codes = set(codes)
        '''

        for i in range(0, len(codes)):
            code = codes[i]
            rank = ranks.setdefault(code, [])
            rank.append(i + 1)

    data = []
    codes = ranks.keys()
    codes = list(codes)
    codes.sort()

    for code in codes:
        print code, np.mean(ranks[code]), np.std(ranks[code]), ranks[code]
        data.append([code, np.mean(ranks[code]), np.std(ranks[code]), ranks[code]])
    
    df = pd.DataFrame(data, index = codes, columns = ['code', 'rank_mean', 'rank_std','ranks'])
    df.to_csv('./tmp/persistence.csv')
