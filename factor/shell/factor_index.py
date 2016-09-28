#coding=utf8


import pandas as pd


def beta_index(beta_df):

    dates = beta_df.index
    codes = beta_df.columns

    for d in dates:
        betas = beta_df.loc[d]
        print betas

    return 1


if __name__ == '__main__':
    beta_df = pd.read_csv('./tmp/beta.csv', index_col = 'date', parse_dates = ['date'])
    #print beta_df
    beta_index(beta_df)
