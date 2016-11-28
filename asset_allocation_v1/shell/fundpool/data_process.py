#coding=utf8


import pandas as pd
import numpy  as np
from datetime import datetime


if __name__ == '__main__':


    fund_df = pd.read_csv('./data/tq_fd_basicinfo', index_col = 'SECURITYID', parse_dates = ['FOUNDDATE'])

    found_date_cols    = ['FOUNDDATE']
    fund_found_date_df = fund_df[found_date_cols]
    fund_found_date_df = fund_found_date_df[fund_found_date_df['FOUNDDATE'] <= datetime.strptime('2013-10-01','%Y-%m-%d')]
    fund_info_df       = fund_df.loc[fund_found_date_df.index]

    fund_info_df.reset_index(inplace = True)
    fund_info_df.set_index(['SECODE'], inplace = True)
    fund_info_cols = ['FDSNAME', 'INVESTSTYLE', 'FOUNDDATE', 'FSYMBOL']
    fund_info_df = fund_info_df[fund_info_cols]

    manager_info_df = pd.read_csv('./data/tq_fd_managerinfo.csv')
    manager_info_secode_df = manager_info_df['SECODE']
    manager_info_secode_df.dropna(inplace = True)
    manager_info_df = manager_info_df.loc[manager_info_secode_df.index]
    manager_info_index_df = manager_info_df['SECODE'].astype(int)
    del manager_info_df['SECODE']
    manager_info_df['SECODE'] = manager_info_index_df
    manager_info_df = manager_info_df.reset_index()
    manager_info_df = manager_info_df.set_index('SECODE')
    manager_info_df = manager_info_df[manager_info_df['ISINCUMBENT'] == 1]

    #print manager_info_df.head()
    #manager_info_cols = ['PSCODE','PSNAME','GENDER','BIRTHDATE','DEGREE','JOBTITLE','REMARK']
    manager_info_cols = ['PSCODE','PSNAME','JOBTITLE','REMARK']
    manager_info_df = manager_info_df[manager_info_cols]
    manager_info_df['PSCODE'] = manager_info_df['PSCODE'].astype(int)
    manager_info_df = manager_info_df.loc[fund_info_df.index]

    manager_info_df.reset_index(inplace = True)
    fund_info_df.reset_index(inplace=True)

    manager_fund_info_df = pd.merge(manager_info_df, fund_info_df, on = 'SECODE')
    manager_fund_info_df.reset_index()
    #manager_fund_info_df.set_index(['SECODE'], inplace=True)
    manager_fund_info_df.to_csv('manager_fund_info.csv')

    managersta_df = pd.read_csv('./data/tq_fd_managersta')
    managersta_cols = ['PSCODE', 'BIRTH', 'GENDER', 'TOTYEARS', 'CURFCOUNT', 'DEGREE']
    managersta_df = managersta_df[managersta_cols]
    manager_fund_info_df = pd.merge(manager_fund_info_df, managersta_df, on = 'PSCODE')
    #print manager_fund_info_df

    manager_performance_df = pd.read_csv('./data/tq_fd_mgperformance')
    #manager_performance_df.columns
    #manager_performance = manager_performance.loc[fund_found_date_df.index]
    manager_performance_df_cols = ['SECODE', 'MANAGERCODE', 'MANAGERNAME', 'PCHG', 'TENUREYIELDYR', 'ALLRANK', 'ALLAVGYIELD', 'CLASSRANK']
    manager_performance_df = manager_performance_df[manager_performance_df_cols]
    manager_performance_df.reset_index(inplace=True)
    manager_performance_df.rename_axis({'MANAGERCODE': 'PSCODE'}, axis = 'columns', inplace = True)
    manager_performance_df['PSCODE'] = manager_performance_df['PSCODE'].astype(int)
    manager_fund_info_df.to_csv('manager_fund_info.csv')
    #print manager_fund_info_df['BIRTHDATE']
    #manager_performance_df.to_csv('manager_performance.csv')

    #print manager_fund_info_df.head()
    #print manager_performance_df.head()
    manager_fund_performance_df = pd.merge(manager_fund_info_df, manager_performance_df, on = ['PSCODE', 'SECODE'])
    manager_fund_performance_df.set_index(['SECODE'], inplace=True)
    #print manager_fund_performance_df
    manager_fund_performance_df.to_csv('manager_fund_performance.csv')
    #manager_info = manager_sta.loc[manager_performance['MANAGERCODE'].values]
    #print manager_info
    #print manager_performance[manager_performance_cols]
