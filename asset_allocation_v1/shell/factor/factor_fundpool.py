#coding=utf8



import pandas as pd
import sys
sys.path.append('shell')
import config
import DBData
import statsmodels.api as sm
import numpy as np
import time



if __name__ == '__main__':


    #industry_df = pd.read_csv('./data/industry_index.csv', index_col = ['date'], parse_dates = ['date'])
    industry_df = pd.read_csv('./data/factor_index.csv', index_col = ['date'], parse_dates = ['date'])
    #industry_dfr = industry_df.pct_change().fillna(0.0)
    #industry_dfr = industry_df.fillna(0.0)
    cols = ['beta', 'market_value', 'momentum', 'dastd', 'bp', 'liquidity']
    industry_dfr = industry_df[cols]
    #print industry_dfr
    #print industry_dfr
    dates = industry_dfr.index
    dates = dates[-956 : ]
    #print dates
    back = 240
    interval = 60
    for i in range(2 * back , len(dates)):

        if i % 60 == 0:

            start_date = dates[i - 2 * back]
            end_date = dates[i]
            df = DBData.stock_day_fund_value(start_date, end_date)
            dfr= df.pct_change().fillna(0.0)

            fund_params = {}
            for j in range(back , len(dfr)):

                tmp_dfr = dfr.iloc[ j - back : j]
                tmp_dates = tmp_dfr.index
                tmp_industry_dfr = industry_dfr.loc[tmp_dates]
                date = dfr.index[j]

                X = tmp_industry_dfr.values
                X = sm.add_constant(X)
                y = tmp_dfr.values
                model  = sm.OLS(y,X)
                result = model.fit()

                #print date, len(result.params)
                params_m = np.matrix(result.params)
                params_m = params_m.T
                for n in range(0, len(dfr.columns)):
                    code = dfr.columns[n]
                    paras = params_m[n]
                    fund_params.setdefault(code, []).append(list(paras.getA()[0]))

            #print fund_params
            code_params = []
            codes = []
            for code in fund_params.keys():
                params = fund_params[code]
                params_df = pd.DataFrame(params)
                stab = np.mean(params_df) / np.std(params_df)
                #print code, stab.values
                code_params.append(stab.values)
                codes.append(code)

            code_params_df = pd.DataFrame(code_params, index = codes)
            #print code_params_df

            fundpool_codes = []
            for p_index in range(1, 7):
                tmp_params = code_params_df[p_index].copy()
                tmp_params.sort_values(inplace = True, ascending=False)
                for n in range(0, 10):
                    fundpool_codes.append(tmp_params.index[n])
                for n in range(1, 11):
                    fundpool_codes.append(tmp_params.index[-1 * n])
                #fundpool_codes.append(tmp_params.index[0])
                #fundpool_codes.append(tmp_params.index[-1])

            print end_date.strftime('%Y-%m-%d'),
            for code in fundpool_codes:
                print code,
            print

    #df = DBData.stock_day_fund_value(start_date, end_date)
    #df.to_csv('./data/test_fund.csv')
    #fund_df = pd.read_csv('./data/test_fund.csv', index_col = ['date'], parse_dates = ['date'])
    #fund_dfr = fund_df.pct_change().fillna(0.0)
    #dates = fund_dfr.index & industry_dfr.index
    #back = 252
    #industry_dfr = industry_dfr.loc[dates]
    #fund_dfr = fund_dfr.loc[dates]


    #print industry_dfr.index
    #print dfr.index
    '''
    allfund_results = []
    allparams_results = []
    for col in fund_dfr.columns:

        dfr = fund_dfr[[col]]

        results = []
        params = []
        ds = []
        total = 0
        correct = 0
        bias = 0
        rsquared_adj_sum = 0
        n = 0
        for i in range(back, len(dates) - 1):
            d = dates[i]
            industry_r = industry_dfr.iloc[i - back : i]
            #print industry_r.values
            fund_r     = dfr.iloc[i - back : i]
            #tmpdf = pd.concat([industry_r, fund_r], axis = 1)
            #print tmpdf
            #print industry_r
            #print fund_r

            #print fund_r.values
            X = industry_r.values
            X = sm.add_constant(X)
            y = fund_r.values
            #print X, y
            model  = sm.OLS(y,X)
            result = model.fit()
            #print d
            #print result.summary()
            #print result.params

            rsquared_adj = result.rsquared_adj
            industry_r = np.append(np.array(1), industry_dfr.iloc[i].values)
            #industry_r = industry_dfr.iloc[i].values
            fund_r = dfr.iloc[i].values
            coef = result.params
            real = fund_r
            predict = np.dot(industry_r, coef)
            if real >= 0 and predict >= 0:
                correct += 1
            elif real <= 0 and predict <= 0:
                correct += 1
            total += 1
            bias += (predict - real[0]) ** 2
            ds.append(d)
            results.append([predict, real[0], rsquared_adj])
            params.append(coef)
            #print d, predict, real[0], rsquared_adj
            rsquared_adj_sum += rsquared_adj

        result_df = pd.DataFrame(results, index = ds, columns = ['predict', 'real', 'r2'])
        params_df = pd.DataFrame(params, index = ds)
        params_disperse_df = np.mean(params_df) / np.std(params_df)
        allparams_results.append(params_disperse_df.values)
        allfund_results.append([1.0 * correct / total, (bias / total) ** 0.5, rsquared_adj_sum / total])
        print col, 'done'

    allfunddf = pd.DataFrame(allfund_results, index = fund_dfr.columns, columns = ['correct_rate','bias','rsquared_adj'])
    allfunddf.index.name = 'code'
    allfunddf.to_csv('allfundstylefactor.csv')
    allparams_results_df = pd.DataFrame(allparams_results, index=fund_dfr.columns)
    print allparams_results_df
    allparams_results_df.to_csv('allparams_results.csv')
    '''
