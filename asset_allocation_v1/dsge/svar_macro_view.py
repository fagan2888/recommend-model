from rpy2 import robjects
from rpy2 import rinterface as ri
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, data
import os
import numpy as np
import pandas as pd

robjects.r('''
        train_mgarch <- function(data){
            library(rugarch)
            library(rmgarch)
            garch11.spec = ugarchspec(mean.model = list(armaOrder = c(1,1)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"), distribution.model = "norm")
            dcc.garch11.spec = dccspec(uspec = multispec(replicate(5, garch11.spec)), dccOrder = c(1,1), distribution = "mvnorm")
            dcc.fit = dccfit(dcc.garch11.spec, data = data)
            dcc.fcst = dccforecast(dcc.fit, n.ahead = 60)

            pre_corr = dcc.fcst@mforecast$H
            #print(cov(data[nrow(data)-200:nrow(data),]))
            ul_pre_corr = unlist(pre_corr)
            ul_pre_corr = ul_pre_corr[(length(ul_pre_corr)-ncol(data)**2 + 1):length(ul_pre_corr)]
            return(ul_pre_corr)
            }
            ''')

robjects.r('''
        train_var <- function(data) {
            library(vars)
            data(Canada)
            var2c = VAR(Canada, p = 2, type = 'const')

            var2c_ser = restrict(var2c, method = 'ser', thresh = 2)
            #print(var2c_ser$restrictions)
            #summary(var2c_ser)

            var_f10 = predict(var2c_ser, n.ahead = 10, ci = 0.95)
            return(var_f10)
            }
            ''')


def train_mgarch(data):
    r_train_mgarch = robjects.r['train_mgarch']
    res = r_train_mgarch(data)
    n_assets = data.shape[1]
    mgarch_res = r_train_mgarch(data)
    mgarch_res = np.array(mgarch_res).reshape(n_assets, n_assets)
    return(res)

def generate_multi_asset_data():
    sh300 = pd.read_csv('./data/sh300.csv', index_col = 0, parse_dates = True)['close']
    zz500 = pd.read_csv('./data/zz500.csv', index_col = 0, parse_dates = True)['close']
    sp500 = pd.read_csv('./data/sp500.csv', index_col = 0, parse_dates = True)['close']
    hsi = pd.read_csv('./data/hsi.csv', index_col = 0, parse_dates = True)['close']
    gold = pd.read_csv('./data/gold.csv', index_col = 0, parse_dates = True)['close']

    df = pd.concat([sh300, zz500, sp500, hsi, gold], 1).pct_change().dropna()
    df.columns = ['sh300', 'zz500', 'sp500', 'hsi', 'gold']
    return df

if __name__ == '__main__':
    numpy2ri.activate()
    assets = generate_multi_asset_data()
    assets_arr = assets.values[-250:]
    mgarch_res = train_mgarch(assets_arr)
    print mgarch_res
