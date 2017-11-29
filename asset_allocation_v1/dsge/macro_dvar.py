import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
from statsmodels.tsa.base.datetools import dates_from_str
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def generate_data():
    cpi = pd.read_csv('data/cpi.csv', index_col=0, parse_dates=True).resample('m').last()
    bond = pd.read_csv('data/bond.csv', index_col=0, parse_dates=True)
    bond = bond.replace(0.0, np.nan)
    bond = bond.resample('m').last()

    m2 = pd.read_csv('data/m2.csv', index_col=0, parse_dates=True).resample('m').last()
    m2 = np.log(m2)

    nci = pd.read_csv('data/nci.csv', index_col=0, parse_dates=True).resample('m').last()
    df = pd.concat([cpi, bond, m2], 1).dropna()
    df.fillna(method='pad', inplace=True)
    #df.plot()
    #plt.show()
    return df

def train(data, fr):
    model = VAR(data)
    results = model.fit(12)
    # results = model.fit(maxlags=15, ic = 'aic')
    # results.plot_acorr()
    # model.select_order(15)
    lag_order = results.k_ar
    forecast = results.forecast(data.values[-lag_order:], fr)
    # print data.values[-lag_order:]
    # results.plot_forecast(6)
    # plt.show()
    # irf = results.irf(10)
    # irf.plot(impulse='realgdp')
    # irf.plot_cum_effects(orth=False)
    # plt.show()
    forecast = forecast[-1, :]
    return forecast

def train_dvar(data, step = 1):
    var = DynamicVAR(data, lag_order=1, window_type='expanding')
    var.plot_forecast(step)
    plt.show()

def predict(data, fr, tn):
    dates = data.index
    columns = data.columns
    tm = dates[tn:-fr] #this month
    nm = dates[tn+fr:] #next month
    pres = []
    for date in tm:
        pre = train(data.loc[:date],fr)
        pres.append(pre)

    result_df = pd.DataFrame(data = pres, index=nm, columns=columns)
    print result_df

    data = data.loc[nm]

    for col in columns:
        plt.figure(figsize=(25, 15))
        plt.title(str(col))
        plt.plot(data.loc['2007':, col], color = 'b', label = 'real %s'%col)
        plt.plot(result_df.loc['2007':, col], color = 'r', label = '%s predicted %d months ago'%(col, fr))
        plt.legend(loc = 1)
        plt.show()


def handle():
    tn = 30 # use at least 30 data to train
    fr = 6 # forecast 6 months later
    data = generate_data()
    # train(data, fr)
    predict(data,fr,tn)

if __name__ == '__main__':
    handle()
