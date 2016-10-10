#coding=utf8


import pandas as pd
import numpy as np
import datetime
import calendar



def get_date_df(df, start_date, end_date):
     _df = df[df.index <= datetime.datetime.strptime(end_date,'%Y-%m-%d').date()]
     _df = _df[_df.index >= datetime.datetime.strptime(start_date,'%Y-%m-%d').date()]
     return _df



def last_friday():
    date   = datetime.date.today()
    oneday = datetime.timedelta(days=1)

    while date.weekday() != calendar.FRIDAY:
        date -= oneday

    date = date.strftime('%Y-%m-%d')
    return date

def portfolio_nav(df_inc, df_position) :
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

     df = df_inc[start_date:]

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
     df_result.insert(0, 'portfolio', df_result.sum(axis=1))               

     return df_result

if __name__ == '__main__':


    df_inc  = pd.read_csv('./testcases/portfolio_nav_inc_df.csv', index_col = 'date', parse_dates = ['date'] )
    df_position = pd.read_csv('./testcases/portfolio_nav_position_df.csv', index_col = 'date', parse_dates = ['date'] )

    print portfolio_nav(df_inc, df_position)


