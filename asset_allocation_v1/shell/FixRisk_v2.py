#coding=utf8


import pandas as pd
import numpy  as np
import sys
sys.path.append('shell')
import portfolio
import datetime
import random


if __name__ == '__main__':


    #code         = '000011.OF'
    #code         = '000300.SH'
    interval     = 20
    short_period = 20
    long_period  = 120


    #df = pd.read_csv('./data/000905.csv', index_col='date', parse_dates=['date'])
    alldf = pd.read_csv('./data/index.csv', index_col='date', parse_dates=['date'])
    #df = pd.read_csv('./data/166002-160505-260110-000011.csv', index_col='date', parse_dates=['date'])
    #df = pd.read_csv('./data/index.csv', index_col='date', parse_dates=['date'])
    #df = pd.read_csv('./data/000905-000016-000300.csv', index_col='date', parse_dates=['date'])
    ##df = pd.read_csv('./data/166002-160505.csv', index_col='date', parse_dates=['date'])
    #df = pd.read_csv('./data/166002-160505-260110-000011.csv', index_col='date', parse_dates=['date'])
    #df = pd.read_csv('./data/spgoldhs.csv', index_col='date', parse_dates=['date'])
    #df = pd.read_csv('./data/guan_fund_value.csv', index_col='date', parse_dates=['date'])
    #df = pd.read_csv('./data/labelasset.csv', index_col='date', parse_dates=['date'])
    #cols = df.columns
    #code = cols[10]

    alldfr = alldf.pct_change().fillna(0.0)


    for code in alldf.columns[0:1]:

        #code = '000905.SH'
        #code = 'largecap'
        #code = 'smallcap'
        #code = 'growth'
        #code = 'value'
        #code = 'rise'
        #code = 'decline'
        #code = 'oscillation'

        df = alldf.fillna(method='pad')
        df = df[[code]]

        df['rs_r20'] = df[code].rolling(window = interval).apply(lambda x: x[-1] / x[0] - 1)

        df.dropna(inplace=True)
        df['rs_return'] = df['rs_r20'].rolling(window=short_period).mean() # r
        df['rs_risk'] = df['rs_r20'].rolling(window=short_period).std()   # risk
        df.dropna(inplace=True)

        dates = df.index

        vs    = [1]
        ovs   = [1]
        ds    = [dates[long_period]]
        ps    = [0]
        pds   = [dates[long_period]]
        #risk_positions = [0]

        position     = 0

        for i in range(long_period, len(dates) - 2):

            d = dates[i]

            long_start_date = df.index[i - long_period]
            short_start_date = df.index[i - short_period]
            end_date = df.index[i]

            risk = df.loc[end_date, 'rs_risk']
            r    = df.loc[end_date, 'rs_return']

            train_df = df.loc[long_start_date: end_date,]
            tmp_df = train_df.copy(deep = True)

            l = len(tmp_df)
            for n in range(0, l):
                tmp_df.iloc[l - 1 - n] = tmp_df.iloc[l - 1 - n] * (0.5 ** ( 1.0 * n / long_period))

            look_back = long_period
            loop_num = look_back / 2
            randoms = []
            rep_num = loop_num * (look_back / 2) / look_back
            day_indexs = range(0, look_back) * rep_num
            random.shuffle(day_indexs)

            day_indexs = np.array(day_indexs)
            day_indexs = day_indexs.reshape(len(day_indexs) / (look_back / 2), look_back / 2)


            print d
            tmp_positions = []

            for index in day_indexs:

                tdf = tmp_df.iloc[index]

                riskmean = tdf['rs_risk'].mean()
                riskstd  = tdf['rs_risk'].std()

                rmean = tdf['rs_return'].mean()
                rstd  = tdf['rs_return'].std()

                position = 0

                if risk >= riskmean + 2 * riskstd:
                    position = 0.0
                elif risk <= riskmean:
                    position = 1.0
                else:
                    position = riskmean / risk

                tmp_positions.append(position)

            position = np.array(tmp_positions).mean()

            if position == 0 or position == 1.0:
                pass
            elif position <= ps[-1] * 0.5 or (position >= ps[-1] * 2 and ps[-1] > 0) or abs(position - ps[-1]) >= 0.2:
                pass
            else:
                position = ps[-1]


            r = alldfr.loc[dates[i + 1], code]
            #r = dfr.loc[dates[i + 2], code]
            v = vs[-1] * (1 + r * position)
            ovs.append(ovs[-1] * (1 + r))
            #v = vs[-1] * (1 + r)

            vs.append(v)
            ds.append(dates[i + 1])
            ps.append(position)
            pds.append(dates[i + 1])


        vdf = pd.DataFrame(np.matrix([vs, ovs]).T, index=ds, columns=['nav', code])
        #vdf = pd.DataFrame(vs, index=ds, columns=[code])
        vdf.index.name = 'date'

        #print vdf
        pdf = pd.DataFrame(ps, index=pds, columns=[code])
        pdf.index.name = 'date'

        #ma_df = pd.DataFrame(mas, index = mads, columns = ['ma'])
        #start_date = datetime.datetime.strptime('2010-01-01','%Y-%m-%d')
        #vdf  = vdf[vdf.index >= start_date]
        #vdf  = vdf / vdf.iloc[0, :]
        #vdf = vdf.resample('M')
        vdf.to_csv('./vdf.csv')
        pdf.to_csv('./pdf.csv')


        print 'fixed risk'
        print "sharpe : ", portfolio.portfolio_sharpe(vdf['nav'],'d')
        print "annual_return : ", portfolio.portfolio_return(vdf['nav'],'d')
        print "maxdrawdown : ", portfolio.portfolio_maxdrawdown(vdf['nav'])
        print "risk : ", np.std(vdf['nav'].pct_change().fillna(0).values) * (252 ** 0.5)

        print
        print code
        print "sharpe : ", portfolio.portfolio_sharpe(vdf[code],'d')
        print "annual_return : ", portfolio.portfolio_return(vdf[code],'d')
        print "maxdrawdown : ", portfolio.portfolio_maxdrawdown(vdf[code])
        print "risk : ", np.std(vdf[code].pct_change().fillna(0).values) * (252 ** 0.5)

        presult = pdf.rolling(window = 2, min_periods = 1).apply(lambda x : x[1] - x[0] if len(x) > 1 else x[0])
        presult = presult.abs().sum(axis = 1).to_frame('turnover').sum()
        print presult

        #print code
        #print "sharpe : ", portfolio.portfolio_sharpe(vdf[code],'d')
        #print "annual_return : ", portfolio.portfolio_return(vdf[code],'d')
        #print "maxdrawdown : ", portfolio.portfolio_maxdrawdown(vdf[code])
        #print "risk : ", np.std(vdf[code].pct_change().fillna(0).values) * (252 ** 0.5)

        #print
        #print
        #print

        #print vdf.corr()
    #print allvdf
