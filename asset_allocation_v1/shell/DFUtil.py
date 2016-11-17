#coding=utf8


import pandas as pd
import numpy as np
import datetime
import calendar



def get_date_df(df, start_date, end_date):
    _df = df[df.index <= end_date]
    _df = _df[_df.index >= start_date]
    return _df



def last_friday():
    date   = datetime.date.today()
    oneday = datetime.timedelta(days=1)

    while date.weekday() != calendar.FRIDAY:
       date -= oneday

    date = date.strftime('%Y-%m-%d')
    return date

def portfolio_nav(df_inc, df_position, result_col='portfolio') :
    '''calc nav for portfolio
    '''
    #
    # 从第一次调仓开始算起.
    #
    # [XXX] 调仓日的处理是先计算收益,然后再调仓, 因为在实际中, 调仓的
    # 动作也是在收盘确认之后发生的
    #
    start_date = df_position.index.min()
    
    if start_date not in df_inc.index:
        df_inc.loc[start_date] = 0
        df_inc.sort_index(inplace=True)

    df = df_inc[start_date:]

    df_position['cash'] = 1 - df_position.sum(axis=1)
    df_inc['cash'] = 0.0

    assets_s = pd.Series(np.zeros(len(df.columns)), index=df.columns) # 当前资产
    assets_s[0] = 1

    #
    # 计算每天的各资产持仓情况
    #
    df_result = pd.DataFrame(index=df.index, columns=df.columns)
    for day,row in df.iterrows():
        # 如果不是第一天, 首先计算当前资产收益
        if day != start_date:
            assets_s = assets_s * (row + 1)

        # 如果是调仓日, 则调仓
        if day in df_position.index: # 调仓日
            assets_s = assets_s.sum() * df_position.loc[day]

        # 日末各个基金持仓情况
        df_result.loc[day] = assets_s

    #
    # 计算资产组合净值
    #
    df_result.insert(0, result_col, df_result.sum(axis=1))               

    return df_result

def load_nav_csv(csv, columns=None, reindex=None):
    if columns and 'date' not in columns:
        columns.insert(0, 'date')
        
    df = pd.read_csv(csv, index_col='date', parse_dates=['date'], usecols=columns)

    if reindex:
        #
        # 重索引的时候需要, 需要首先用两个索引的并集,填充之后再取
        # reindex, 否则中间可能会造成增长率丢失
        # 
        index = df.index.union(reindex)
        df = df.reindex(index, method='pad').reindex(reindex)
        
    return df

def load_inc_csv(csv, columns=None, reindex=None):
    if columns and 'date' not in columns:
        columns.insert(0, 'date')
    #
    # [XXX] 不能直接调用load_nav_csv, 否则会造成第一行的增长率丢失
    #
    df = pd.read_csv(csv, index_col='date', parse_dates=['date'], usecols=columns)

    if reindex is not None:
        #
        # 重索引的时候需要, 需要首先用两个索引的并集,填充之后再计算增长率
        # 
        index = df.index.union(reindex)
        df = df.reindex(index, method='pad')

    # 计算增长率
    dfr = df.pct_change().fillna(0.0)

    if reindex is not None:
        dfr = dfr.reindex(reindex)
        
    return dfr

def pad_sum_to_one(df, by, ratio='ratio'):
    df3 = df[ratio].groupby(by).agg(['sum', 'idxmax'])
    df.ix[df3['idxmax'], ratio] += (1 - df3['sum']).values

    print df[ratio].groupby(by).sum()
    
    return df

def filter_by_turnover_rate(df, turnover_rate):
    df_result = pd.DataFrame(columns=['risk', 'date', 'category', 'fund', 'ratio'])
    for k0, v0 in df.groupby('risk'):
        df_tmp = filter_by_turnover_rate_per_risk(v0, turnover_rate)
        if not df_tmp.empty:
            df_result = pd.concat([df_result, df_tmp])
            
    return df_result

def filter_by_turnover_rate_per_risk(df, turnover_rate):
    df_result = pd.DataFrame(columns=['risk', 'date', 'category', 'fund', 'ratio'])
    df_last=None
    for k1, v1 in df.groupby(['risk', 'date']):
        if df_last is None:
            df_last = v1[['category', 'fund','ratio']].set_index(['category', 'fund'])
            df_result = pd.concat([df_result, v1])
        else:
            df_current = v1[['category', 'fund', 'ratio']].set_index(['category', 'fund'])
            df_diff = df_current - df_last
            xsum = df_diff['ratio'].abs().sum()
            if df_diff.isnull().values.any() or xsum >= turnover_rate:
                df_result = pd.concat([df_result, v1])
                df_last = df_current
                
    return df_result

def portfolio_import(df):
    pass;


def categories_types(as_int=False):
    if as_int:
        return {
            'largecap'        : 11, # 大盘
            'smallcap'        : 12, # 小盘
            'rise'            : 13, # 上涨
            'oscillation'     : 14, # 震荡
            'decline'         : 15, # 下跌
            'growth'          : 16, # 成长
            'value'           : 17, # 价值

            'ratebond'        : 21, # 利率债
            'creditbond'      : 22, # 信用债
            'convertiblebond' : 23, # 可转债

            'money'           : 31, # 货币

            'SP500.SPI'       : 41, # 标普
            'GLNC'            : 42, # 黄金
            'HSCI.HI'         : 43, # 恒生
        }
        
    #
    # 输出配置数据
    #
    return {
        'largecap'        : '11', # 大盘
        'smallcap'        : '12', # 小盘
        'rise'            : '13', # 上涨
        'oscillation'     : '14', # 震荡
        'decline'         : '15', # 下跌
        'growth'          : '16', # 成长
        'value'           : '17', # 价值

        'ratebond'        : '21', # 利率债
        'creditbond'      : '22', # 信用债
        'convertiblebond' : '23', # 可转债

        'money'           : '31', # 货币

        'SP500.SPI'       : '41', # 标普
        'GLNC'            : '42', # 黄金
        'HSCI.HI'         : '43', # 恒生
    }


if __name__ == '__main__':


    df_inc  = pd.read_csv('./testcases/portfolio_nav_inc_df.csv', index_col = 'date', parse_dates = ['date'] )
    df_position = pd.read_csv('./testcases/portfolio_nav_position_df.csv', index_col = 'date', parse_dates = ['date'] )

    print portfolio_nav(df_inc, df_position)


