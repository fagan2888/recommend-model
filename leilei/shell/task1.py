#coding=utf8


import pandas as pd


df = pd.read_csv('./data/df.csv', index_col = 'date', parse_dates = ['date'])


#backs = [252, 252 * 2, 252 * 3, 252 * 4, 252 * 5]
#dates = df.index


'''
for code in df.columns:
    for back in backs:
        n = 0
        sum_r = 0
        for i in range(back, len(dates)):
            d = dates[i]
            back_d = dates[i - back]
            r = df.loc[d, code] / df.loc[back_d, code] - 1
            sum_r = sum_r + r
            n = n + 1
        print code, back, sum_r / n
'''


df = df.resample('M').last()

dates = df.index
backs = [12, 12 * 2, 12 * 3, 12 * 4, 12 * 5]
for code in df.columns:
    for back in backs:
        sumr = 0
        n = 0
        for i in range(back, len(dates)):
            rs = 0
            n = n + 1
            start_d = dates[ i - back]
            for j in range(0, back):
                d = dates[i - j]
                r = df.loc[d, code] /  df.loc[start_d, code] - 1
                rs = rs + r
            sumr = sumr + rs
        sumr = sumr / back
        print code, back, sumr / n
