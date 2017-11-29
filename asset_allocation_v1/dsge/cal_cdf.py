#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import pandas as pd
from scipy import stats

def load_data(path):
    df = pd.read_csv(path, index_col = 0, parse_dates = True)
    return df

def cal_cdf(data):
    cdfs = []
    for i in range(len(data)):
        v = float(data.values[i,0])
        m = float(data.values[i,1])
        s = float(data.values[i,2])
        print v,m,s
        tmp_cdf = stats.norm.cdf(v, m, s)
        cdfs.append(tmp_cdf)
    data['cdf'] = cdfs
    data.to_csv('sh300_prob.csv', index_label = 'date')
    print data

def cal_cdf_t(data):
    cdfs = []
    for i in range(len(data)):
        v = float(data.values[i,0])
        m = float(data.values[i,1])
        s = float(data.values[i,2])
        print v,m,s
        tmp_cdf = stats.t.cdf(v, 2, m, s)
        cdfs.append(tmp_cdf)
    data['cdf'] = cdfs
    data.to_csv('zz500_prob_t.csv', index_label = 'date')
    print data
    
if __name__ == '__main__':
    path = 'view/zz500_mean_std_view.csv'
    data = load_data(path)
    cal_cdf_t(data)
