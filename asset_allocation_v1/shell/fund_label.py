from sqlalchemy import MetaData, Table, select
from db import database
from db import base_ra_fund_nav as load_fund_daily
import pandas as pd
import numpy as np

def load_fund_code():
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_fund', metadata, autoload = True)

    columns = [
            t.c.ra_code.label('code'),
            t.c.ra_type.label('type'),
            ]

    s = select(columns).where(t.c.ra_type == 1)

    df = pd.read_sql(s, db)

    return df

def label(code, df1, df2):
    fund_nav = load_fund_daily.load_daily('2005-01-04', '2018-08-17', codes = code)
    fund_nav = fund_nav.loc[:'2018-08-17']
    df1 = df1.copy()
    df1 = df1.copy()
    df1 = df1.loc[fund_nav.index]
    df2 = df2.loc[fund_nav.index]
    fund_nav = fund_nav.values.flat[:]
    df1_v = df1.values.flat[:]
    df2_v = df2.values.flat[:]
    fund_nav = np.nan_to_num(fund_nav)
    df1_v = np.nan_to_num(df1_v)
    df2_v = np.nan_to_num(df2_v)
    l1 = np.corrcoef(fund_nav, df1_v)[0,1]
    l2 = np.corrcoef(fund_nav, df2_v)[0,1]
    if (l1 < 0.7) and (l2 < 0.7):
        return 0
    elif l1 > l2:
        return 11101
    else:
        return 11102

def handle():
    #code_df = load_fund_code()
    code_df = pd.read_csv('unknow.csv', encoding = 'gbk')
    code_df = code_df[code_df.type == 11101]
    codes = np.unique(code_df['code'])

    '''
    df = pd.DataFrame()
    fund_code = []
    for code in codes:
        fund_nav = load_fund_daily.load_daily('2005-01-04', '2018-08-17', codes = [code])
        if len(fund_nav) == 0:
            print code
            fund_code.append(code)
    df['fund_code'] = fund_code
    df.to_csv('code.csv')
    '''
    #np.save(file('fund_code.npy', 'w'), code)
    sh300 = pd.read_csv('tmp/120000001_ori_day_data.csv', index_col = 0, parse_dates = True)
    zz500 = pd.read_csv('tmp/120000002_ori_day_data.csv', index_col = 0, parse_dates = True)
    sh300 = sh300['close']
    zz500 = zz500['close']
    #codes = np.load('fund_code.npy')

    count = 0
    codes = codes
    labels = []
    for code in codes:
        code = '%06d'%code
        try:
            labels.append(label([code], sh300, zz500))
        except Exception, e:
            print e
            labels.append(-1)
        count += 1
        print count
    result_df = pd.DataFrame({'labels': labels}, index = codes)
    result_df.to_csv('fund_label_2.csv', index_label = 'ra_code')
    print result_df

if __name__ == '__main__':
    handle()
