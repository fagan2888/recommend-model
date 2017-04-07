#coding=utf8


import pandas as pd


if __name__ == '__main__':

    account_df = pd.read_csv('yingmi_account.csv', index_col = ['ya_uid'])

    trade_df = pd.read_csv('yingmi_portfolio_trade_status.csv', index_col = ['yp_uid'])

    #print trade_df

    df = trade_df[trade_df['yp_trade_type'] == 'P02']

    uids = set(df.index)

    account_df = account_df.loc[uids]

    account_df.to_csv('account.csv')
