import pandas as pd
import numpy as np
from utils import day_2_week_ipo as d2w
from db import asset_trade_dates as load_td
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

def load_data():
    sh300_indic = pd.read_csv('sh300_indic.csv', index_col = 0, parse_dates = True)
    ipo_data = pd.read_csv('ipo_data.csv', index_col = 0, parse_dates = True)
    trade_dates = load_td.load_trade_dates()
    ipo_data = d2w(ipo_data, trade_dates)
    all_data = pd.merge(sh300_indic, ipo_data, left_index = True, right_index = True, \
            how = 'left')
    all_data = all_data.fillna(0)
    all_data['pct_chg1'] = all_data['pct_chg'].shift(-1)
    all_data = all_data.dropna()
    return all_data

def pca():
    data = load_data()
    std_data = StandardScaler().fit_transform(data.iloc[:, :-1])
    pca = PCA(n_components = 5)
    components = pca.fit_transform(std_data)
    df_components = pd.DataFrame(components, columns = ['c1', 'c2', 'c3', 'c4',\
            'c5'], index = data.index)
    print np.corrcoef(df_components.c1, data.pct_chg1)[0,1]

def predict():
    window = 50
    data = load_data()
    c1_list = []
    c2_list = []
    c3_list = []
    c4_list = []
    c5_list = []
    for i in range(len(data) - window + 1):
        tmp_data = StandardScaler().fit_transform(data.iloc[i: i+window, :-1])
        pca = PCA(n_components = 5)
        components = pca.fit_transform(tmp_data)
        c1, c2, c3, c4, c5 = components[-1, :]
        if i == 0:
            c1_list.append(c1)
            c2_list.append(c2)
            c3_list.append(c3)
            c4_list.append(c4)
            c5_list.append(c5)
        else:
            lc1, lc2, lc3, lc4, lc5 = components[-2, :]
            c1_list.append(c1_list[-1]+c1-lc1)
            c2_list.append(c2_list[-1]+c2-lc2)
            c3_list.append(c3_list[-1]+c3-lc3)
            c4_list.append(c4_list[-1]+c4-lc4)
            c5_list.append(c5_list[-1]+c5-lc5)
    data = data[window-1:]
    data['c1'] = c1_list
    data['c2'] = c2_list
    data['c3'] = c3_list
    data['c4'] = c4_list
    data['c5'] = c5_list
    data.to_csv('sh300_gmx.csv')
    print data

def res_sta():
    data = pd.read_csv('sh300_gmx.csv')
    for i in range(1, 6):
        corr_close = np.corrcoef(data['c%d'%i], data['close'])[0,1]
        corr_pct_chg = np.corrcoef(data['c%d'%i], data['pct_chg'])[0,1]
        corr_pct_chg1 = np.corrcoef(data['c%d'%i], data['pct_chg1'])[0,1]
        rcorr_close = spearmanr(data['c%d'%i], data['close']).correlation
        rcorr_pct_chg = spearmanr(data['c%d'%i], data['pct_chg']).correlation
        rcorr_pct_chg1 = spearmanr(data['c%d'%i], data['pct_chg1']).correlation
        print '###############################'
        print 'c%d'%i
        print corr_close
        print corr_pct_chg
        print corr_pct_chg1
        print rcorr_close
        print rcorr_pct_chg
        print rcorr_pct_chg1

if __name__ == '__main__':
    #predict()
    res_sta()
