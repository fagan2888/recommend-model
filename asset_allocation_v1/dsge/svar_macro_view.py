from rpy2 import robjects
from rpy2.robjects import numpy2ri
#from rpy2 import rinterface as ri
#from rpy2.robjects.packages import importr, data
#import json
import os
import cPickle
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

robjects.r('''
        train_mgarch <- function(data, fn){
            library(parallel)
            library(rugarch)
            library(rmgarch)
            garch11.spec = ugarchspec(mean.model = list(armaOrder = c(1,1)), \
            variance.model = list(garchOrder = c(1,1), model = "sGARCH"), distribution.model = "norm")
            dcc.garch11.spec = dccspec(uspec = multispec(replicate(5, garch11.spec)), \
            dccOrder = c(1,1), distribution = "mvnorm")

            cl = makePSOCKcluster(16)
            dcc.fit = dccfit(dcc.garch11.spec, data = data, cluster = cl)
            dcc.fcst = dccforecast(dcc.fit, n.ahead = fn)

            pre_corr = dcc.fcst@mforecast$H[[1]][,,fn]
            #print(cov(data[nrow(data)-200:nrow(data),]))
            #ul_pre_corr = unlist(pre_corr)
            #ul_pre_corr = ul_pre_corr[(length(ul_pre_corr)-ncol(data)**2 + 1):length(ul_pre_corr)]
            stopCluster(cl)
            return(pre_corr)
            }
            ''')

robjects.r('''
        train_var <- function(data, fn) {
            library(vars)
            #var2c = VAR(data, p = 2, type = 'const')
            var2c = VAR(data, lag.max = 3, ic = 'AIC', type = 'const')

            var2c_ser = restrict(var2c, method = 'ser', thresh = 2)
            var_f = predict(var2c_ser, n.ahead = fn, ci = 0.95)
            #var_f = predict(var2c, n.ahead = fn, ci = 0.95)
            var_f_y1 = var_f$fcst$y1[1]
            var_f_y1_ci = var_f$fcst$y1[4]/2
            var_f_y2 = var_f$fcst$y2[1]
            var_f_y2_ci = var_f$fcst$y2[4]/2
            #var_f_y3 = var_f$fcst$y3[1]
            #var_f_y4 = var_f$fcst$y4[1]
            #var_f_y5 = var_f$fcst$y5[1]
            result = 1:4
            result[1] = var_f_y1
            result[2] = var_f_y1_ci
            result[3] = var_f_y2
            result[4] = var_f_y2_ci
            
            return(result)
            }
            ''')

robjects.r('''
        train_msbvar <- function(data, fn){
            library(MSBVAR)
            data_ts = ts(data, frequency = 12, start = c(2010, 1))
            model = msbvar(data_ts, p = 1, h = 2, lambda0 = 0.8, \
            lambda1 = 0.2, lambda3 = 1, lambda4 = 0.2, \
            lambda5 = 0, mu5 = 0, mu6 = 0, qm = 12, prior = 0)
            model_gibbs = gibbs.msbvar(model, N1 = 1000, N2 = 1000, permute = FALSE, Sigma.idx = 1)
            msforc = forecast(model_gibbs, nsteps = 2, N1 = 1000, N2 = 1000)
            result_mean = apply(msforc$forecast, c(2,3), mean)

            return(result_mean)
          }
           ''')

def train_mgarch(data, fn):
    r_train_mgarch = robjects.r['train_mgarch']
    res = r_train_mgarch(data, fn)
    n_assets = data.shape[1]
    mgarch_res = r_train_mgarch(data, fn)
    mgarch_res = np.array(mgarch_res).reshape(n_assets, n_assets)
    return res

def train_var(data, fn):
    r_train_var = robjects.r['train_var']
    res = r_train_var(data, fn)
    return res

def train_var(data, fn):
    r_train_msbvar = robjects.r['train_msbvar']
    res = r_train_msbvar(data, fn)
    return res

def generate_multi_asset_data():
    #read low-level indicator
    sh300 = pd.read_csv('./data/sh300.csv', index_col = 0, parse_dates = True)['sh300'].pct_change()
    zz500 = pd.read_csv('./data/zz500.csv', index_col = 0, parse_dates = True)['zz500'].pct_change()
    sp500 = pd.read_csv('./data/sp500.csv', index_col = 0, parse_dates = True)['sp500'].pct_change()
    hsi = pd.read_csv('./data/hsi.csv', index_col = 0, parse_dates = True)['hsi'].pct_change()
    gold = pd.read_csv('./data/gold.csv', index_col = 0, parse_dates = True)['gold'].pct_change()
    m2 = pd.read_csv('./data/m2.csv', index_col = 0, parse_dates = True)['m2']
    m2_stock = pd.read_csv('./data/m2_stock.csv', index_col = 0, parse_dates = True)['m2_stock']
    m1 = pd.read_csv('./data/m1.csv', index_col = 0, parse_dates = True)['m1']
    cpi = pd.read_csv('./data/cpi.csv', index_col = 0, parse_dates = True)['cpi']
    bond = pd.read_csv('./data/bond.csv', index_col = 0, parse_dates = True)['bond']
    bond = bond.replace(0.0, np.nan).fillna(method = 'pad')
    sf = pd.read_csv('./data/sf.csv', index_col = 0, parse_dates = True)['sf']
    sf_m2 = (sf - m2_stock.diff()).cumsum().diff(12)
    sf_m2.name = 'sf-m2'
    df = pd.concat([sh300, zz500, sp500, hsi, gold, m2, m2_stock, m1, cpi, bond, sf, sf_m2], 1).fillna(method = 'pad').dropna()

    #calculate high-level indicator
    #leading indicator of house price
    df['bond_inv'] = 1/df['bond']
    #leading indicator of interet rate
    #print sf_m2
    return df

def train_corr_days(data, fn, look_back, start_date = '2010-01-01'):
    dates = data.loc[start_date:].index
    corr_view = {}
    for date in dates:
        tmp_train_data = data.loc[date - timedelta(look_back):date, ['sh300', 'zz500', 'sp500', 'hsi', 'gold']]
        tmp_train_data = tmp_train_data.values

        res = np.array(train_mgarch(tmp_train_data, fn))
        print date, res
        corr_view[date.strftime('%Y-%m-%d')] = res
        if date.day == 8:
            with open('./view/corr_view.json', 'w') as f:
                cPickle.dump(corr_view, f)

def train_mean_days(data, fn, look_back, start_date = '2010-01-01'):
    dates = data.loc[start_date:].index
    mean_view = []
    for date in dates:
        tmp_train_data = \
            data.loc[date - timedelta(look_back):date, \
            ['sh300','zz500', 'sp500','hsi', 'gold', 'bond_inv', 'm1', 'sf-m2', 'cpi']]
        tmp_train_data = tmp_train_data.resample('m').last()
        #tmp_train_data = tmp_train_data[::-20][::-1]
        #tmp_train_data = tmp_train_data.drop_duplicates()
        tmp_train_data = tmp_train_data.values
        tmp_train_data_min = tmp_train_data.min(0)
        tmp_train_data_max = tmp_train_data.max(0)
        tmp_train_data_scaled = (tmp_train_data - tmp_train_data_min)/(tmp_train_data_max-tmp_train_data_min)
        res = np.array(train_var(tmp_train_data_scaled, fn))
        res = res*(tmp_train_data_max[:5] - tmp_train_data_min[:5])+tmp_train_data_min[:5]
        print date, res
        mean_view.append(res)
    df = pd.DataFrame(data = mean_view, index = dates, \
        columns = ['sh300_mean_view', 'zz500_mean_view', 'sp500_mean_view', 'hsi_mean_view', 'gold_mean_view'])
    df.to_csv('./view/asset_mean_view.csv', index_label = 'date')

def train_sh300_mean_days(data, fn, look_back, start_date = '2010-01-01'):
    dates = data.loc[start_date:].index
    sh300_pc = []
    sh300_mean_view = []
    sh300_std_view = []
    zz500_pc = []
    zz500_mean_view = []
    zz500_std_view = []
    for date in dates:
        tmp_train_data = \
            data.loc[date - timedelta(look_back):date, \
            ['sh300', 'zz500', 'hsi', 'bond_inv', 'm1', 'sf-m2', 'cpi']]
        if date.day >= 15:
            tmp_train_data = tmp_train_data.resample('m').last()
        else:
            tmp_train_data = tmp_train_data.resample('m').first()
        #tmp_train_data = tmp_train_data[::-20][::-1]
        #tmp_train_data = tmp_train_data.drop_duplicates()
        tmp_train_data = tmp_train_data.values
        sh300_pc.append(tmp_train_data[-1, 0])
        zz500_pc.append(tmp_train_data[-1, 1])
        tmp_train_data_min = tmp_train_data.min(0)
        tmp_train_data_max = tmp_train_data.max(0)
        tmp_train_data_scaled = (tmp_train_data - tmp_train_data_min)/(tmp_train_data_max-tmp_train_data_min)
        res = np.array(train_var(tmp_train_data_scaled, fn))
#        res = res*(tmp_train_data_max[:4] - tmp_train_data_min[:4])+tmp_train_data_min[:4]
        res[0] = res[0]*(tmp_train_data_max[0] - tmp_train_data_min[0])+tmp_train_data_min[0]
        res[1] = res[1]*(tmp_train_data_max[0] - tmp_train_data_min[0])
        res[2] = res[2]*(tmp_train_data_max[1] - tmp_train_data_min[1])+tmp_train_data_min[1]
        res[3] = res[3]*(tmp_train_data_max[1] - tmp_train_data_min[1])
        sh300_mean_view.append(res[0])
        sh300_std_view.append(res[1])
        zz500_mean_view.append(res[2])
        zz500_std_view.append(res[3])
        print date, res[0], res[1], res[2], res[3]

    df = pd.DataFrame(data = np.column_stack([sh300_pc, sh300_mean_view, sh300_std_view]), index = dates, \
        columns = ['sh300_pc', 'sh300_mean_view', 'sh300_std_view'])
    df.to_csv('./view/sh300_mean_std_view.csv', index_label = 'date')

    df = pd.DataFrame(data = np.column_stack([zz500_pc, zz500_mean_view, zz500_std_view]), index = dates, \
        columns = ['zz500_pc', 'zz500_mean_view', 'zz500_std_view'])
    df.to_csv('./view/zz500_mean_std_view.csv', index_label = 'date')

    #df = pd.DataFrame(data = hsi_mean_view, index = dates, \
    #    columns = ['hsi_mean_view'])
    #df.to_csv('./view/hsi_mean_view.csv', index_label = 'date')

def train_sp500_gold_mean_days(data, fn, look_back, start_date = '2010-01-01'):
    dates = data.loc[start_date:].index
    sp500_mean_view = []
    gold_mean_view = []
    for date in dates:
        tmp_train_data = \
            data.loc[date - timedelta(look_back):date, ['sp500', 'gold']]
        #tmp_train_data = tmp_train_data[::-20][::-1]
        #tmp_train_data = tmp_train_data.drop_duplicates()
        tmp_train_data = tmp_train_data.values
        tmp_train_data_min = tmp_train_data.min(0)
        tmp_train_data_max = tmp_train_data.max(0)
        tmp_train_data_scaled = (tmp_train_data - tmp_train_data_min)/(tmp_train_data_max-tmp_train_data_min)
        res = np.array(train_var(tmp_train_data_scaled, fn))
        res = res*(tmp_train_data_max - tmp_train_data_min)+tmp_train_data_min
        print date, res[0], res[1]
        sp500_mean_view.append(res[0])
        gold_mean_view.append(res[1])
    df = pd.DataFrame(data = sp500_mean_view, index = dates, \
        columns = ['sp500_mean_view'])
    df = pd.DataFrame(data = gold_mean_view, index = dates, \
        columns = ['gold_mean_view'])
    df.to_csv('./view/sp500_mean_view.csv', index_label = 'date')
    df.to_csv('./view/gold_mean_view.csv', index_label = 'date')
'''
def res_sta(data):
    view = pd.read_csv('./view/asset_mean_view.csv', index_col = 0, parse_dates = True)
    data = data.loc[df.index[0]:]
    dates = data.index
    asset_nav = []
    asset_tmp_nav = 1
    for date in dates:
        tmp_view = view.loc[:'dates', 'sh300_mean_view'][-1]
        '''

def train_sh300_mean_msbvar(data, fn, look_back, start_date = '2010-01-01'):
    dates = data.loc[start_date:].index
    mean_view = []
    for date in dates:
        tmp_train_data = \
            data.loc[date - timedelta(look_back):date, \
            ['sh300','zz500', 'bond', 'bond_inv', 'm1', 'sf-m2', 'cpi']]
        tmp_train_data = tmp_train_data.resample('m').last()
        #tmp_train_data = tmp_train_data[::-20][::-1]
        #tmp_train_data = tmp_train_data.drop_duplicates()
        tmp_train_data = tmp_train_data.values
        tmp_train_data_min = tmp_train_data.min(0)
        tmp_train_data_max = tmp_train_data.max(0)
        tmp_train_data_scaled = (tmp_train_data - tmp_train_data_min)/(tmp_train_data_max-tmp_train_data_min)
        res = np.array(train_var(tmp_train_data_scaled, fn))
        print res
        os._exit(0)
        res = res*(tmp_train_data_max[:5] - tmp_train_data_min[:5])+tmp_train_data_min[:5]
        print date, res
        mean_view.append(res)
    df = pd.DataFrame(data = mean_view, index = dates, \
        columns = ['sh300_mean_view', 'zz500_mean_view', 'sp500_mean_view', 'hsi_mean_view', 'gold_mean_view'])
    df.to_csv('./view/asset_mean_view.csv', index_label = 'date')


if __name__ == '__main__':
    numpy2ri.activate()
    FORECAT_DAY = 20
    FORECAT_MONTH = 1
    LOOK_BACK = 365*3
    ASSETS = generate_multi_asset_data()
    #ASSETS.to_csv('./data/assets_indicators.csv', index_label = 'date')
    #os._exit(0)

    #train_mean_days(ASSETS, FORECAT_MONTH, LOOK_BACK)
    #train_sp500_gold_mean_days(ASSETS, FORECAT_DAY, LOOK_BACK)
    #train_corr_days(ASSETS, FORECAT_DAY, LOOK_BACK)
    #train_sh300_mean_msbvar(ASSETS, FORECAT_MONTH, LOOK_BACK)

    train_sh300_mean_days(ASSETS, FORECAT_MONTH, LOOK_BACK)

