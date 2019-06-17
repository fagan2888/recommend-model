#coding=utf8


import getopt
import string
import json
import os
import sys
import logging
sys.path.append('shell')
import click
import config
import pandas as pd
import numpy as np
import LabelAsset
import os
import DBData
import time
import Const
import DFUtil
import LabelAsset
import Financial as fin
from ipdb import set_trace

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_stock_factor, asset_fund_factor, base_fund_fee, asset_ra_pool_fund
from asset import Asset, StockFundAsset
from asset_allocate import MonetaryAllocate
from trade_date import ATradeDate
from monetary_fund_filter import MonetaryFundFilter

import pymysql
import calendar
import numpy as np
import pandas as pd
import re
import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def bond(ctx):
    '''bond fund pool group
    '''
    pass


@bond.command()
@click.pass_context
def low_rank_bond_fund(ctx):


    conn = database.connection('base')
    sql_t = 'select * from ra_fund'
    ra_fund = pd.read_sql(sql=sql_t, con=conn)

    sql_t = 'select ra_code, ra_date, ra_nav_adjusted from ra_fund_nav where ra_date > "2009-01-01"'
    #ra_fund_nav = pd.read_sql(sql=sql_t, con=conn, index_col = ['ra_date','ra_code'])
    #ra_fund_nav = ra_fund_nav.unstack()
    #ra_fund_nav.columns = ra_fund_nav.columns.get_level_values(1)
    #ra_fund_nav.to_csv('./data/ra_fund_nav.csv')
    ra_fund_nav = pd.read_csv('./data/ra_fund_nav.csv', index_col = ['ra_date'])

    sql_t = 'select * from trade_dates'
    td_date = pd.read_sql(sql=sql_t, con=conn)
    tradedate_list = [pd.Timestamp(i) for i in td_date.sort_values(by='td_date').td_date.unique()]


    #Wind db
    conn = pymysql.connect(host='192.168.88.12', user='public', passwd='h76zyeTfVqAehr5J', database='wind', charset='utf8')

    sql_t = 'select * from chinamutualfundassetportfolio'
    #chinamutualfundassetportfolio = pd.read_sql(sql=sql_t, con=conn, index_col = ['OBJECT_ID'])
    #chinamutualfundassetportfolio.to_csv('./data/chinamutualfundassetportfolio.csv')
    chinamutualfundassetportfolio = pd.read_csv('./data/chinamutualfundassetportfolio.csv')

    sql_t = 'select * from chinamutualfundbondportfolio'
    #chinamutualfundbondportfolio = pd.read_sql(sql=sql_t, con=conn, index_col = ['OBJECT_ID'])
    #chinamutualfundbondportfolio.to_csv('./data/chinamutualfundbondportfolio.csv')
    chinamutualfundbondportfolio = pd.read_csv('./data/chinamutualfundbondportfolio.csv')

    sql_t = 'select * from ChinaMutualFundSector'
    #ChinaMutualFundSector = pd.read_sql(sql=sql_t, con=conn, index_col = ['OBJECT_ID'])
    #ChinaMutualFundSector.to_csv('./data/ChinaMutualFundSector.csv')
    ChinaMutualFundSector = pd.read_csv('./data/ChinaMutualFundSector.csv')

    sql_t = 'select * from cbondrating'
    cbondrating = pd.read_sql(sql=sql_t, con=conn)

    sql_t = 'select * from ChinaMutualFundDescription'
    ChinaMutualFundDescription = pd.read_sql(sql=sql_t, con=conn)

    conn.close()


    '''
    2001010301	中长期纯债型基金
    2001010302	短期纯债型基金
    2001010303	混合债券型一级基金
    2001010304	混合债券型二级基金
    2001010305	被动指数型债券基金
    2001010306	增强指数型债券基金
    '''

    #ChinaMutualFundSector_selected_Criteria_List = ['2001010301000000', '2001010302000000', '2001010305000000', '2001010306000000', '2001010303000000', '2001010304000000']
    ChinaMutualFundSector_selected_Criteria_List = ['2001010301000000', '2001010302000000', '2001010305000000', '2001010306000000']
    # TODO: Time series fund type

    ChinaMutualFundSector_selected = ChinaMutualFundSector[ChinaMutualFundSector.S_INFO_SECTOR.isin(ChinaMutualFundSector_selected_Criteria_List)] 
    ChinaMutualFundSector_selected = ChinaMutualFundSector_selected[ChinaMutualFundSector_selected.F_INFO_WINDCODE.map(lambda x: x.split('.')[1]) == 'OF']
    C_type_mutual_fund = ChinaMutualFundDescription[['C' == name[-1] for name in ChinaMutualFundDescription.F_INFO_NAME]]

    ChinaMutualFundSector_selected = ChinaMutualFundSector_selected[ChinaMutualFundSector_selected.F_INFO_WINDCODE.isin(set(C_type_mutual_fund.F_INFO_WINDCODE).intersection(set(ChinaMutualFundSector_selected.F_INFO_WINDCODE)))]

    chinamutualfundassetportfolio_selected_by_asset_type_pool = chinamutualfundassetportfolio[(chinamutualfundassetportfolio.F_PRT_CORPBONDTONAV.fillna(value=0) > 70) & (chinamutualfundassetportfolio.F_PRT_COVERTBONDTONAV.fillna(value=0) < 25)]

    # Selection Criteria II: Asset Allocation criteria e.g. Corporate bond NAV percentage
    chinamutualfundassetportfolio_criteria_1 = chinamutualfundassetportfolio.pivot_table('F_PRT_CORPBONDTONAV', 'F_PRT_ENDDATE', 'S_INFO_WINDCODE')
    chinamutualfundassetportfolio_criteria_2 = chinamutualfundassetportfolio.pivot_table('F_PRT_COVERTBONDTONAV', 'F_PRT_ENDDATE', 'S_INFO_WINDCODE')

    chinamutualfundassetportfolio_selected = chinamutualfundassetportfolio_criteria_1.copy()
    #chinamutualfundassetportfolio_selected = chinamutualfundassetportfolio_selected[chinamutualfundassetportfolio_selected > 80]
    chinamutualfundassetportfolio_selected_ = chinamutualfundassetportfolio_selected.dropna(axis=1, thresh=1)
    chinamutualfundassetportfolio_selected_ = chinamutualfundassetportfolio_selected_.dropna(axis=0, thresh=1)
    # 
    Mutual_Fund_Selected_Pool = chinamutualfundassetportfolio_selected_[list(chinamutualfundassetportfolio_selected_.columns.intersection(ChinaMutualFundSector_selected.F_INFO_WINDCODE))]
    Mutual_Fund_Selected_Pool_ = Mutual_Fund_Selected_Pool.dropna(axis=1, thresh=1)
    Mutual_Fund_Selected_Pool_ = Mutual_Fund_Selected_Pool_.dropna(axis=0, thresh=1)
   
    #Mutual_Fund_Selected_Pool_.columns = Mutual_Fund_Selected_Pool_.columns.map(lambda x: x.split('.')[0])

    # Selection Criteria III: Bond Credit Rating criteria e.g. AAA & A-
    # Special comment: if there is not credit rating or the result as nan, then I set it as the sovereign bond in China 
    # TODO: Time series Credit rating
    chinamutualfundbondportfolio_rating = pd.merge(chinamutualfundbondportfolio, cbondrating[['S_INFO_WINDCODE', 'B_INFO_CREDITRATING']], left_on=['S_INFO_BONDWINDCODE'], right_on=['S_INFO_WINDCODE'], how='left')
    chinamutualfundbondportfolio_rating = chinamutualfundbondportfolio_rating.drop_duplicates()
    chinamutualfundbondportfolio_rating = chinamutualfundbondportfolio_rating[~chinamutualfundbondportfolio_rating.B_INFO_CREDITRATING.isnull()]
    # Add the sum percentage of nav of fixed income of different preference credit rating here
    chinamutualfundbondportfolio_rating_high_credit_list = [ 'AAA', 'AAA+']
    chinamutualfundbondportfolio_rating_lower_credit_list = list(set(list(chinamutualfundbondportfolio_rating.B_INFO_CREDITRATING.drop_duplicates().values)).symmetric_difference(set(chinamutualfundbondportfolio_rating_high_credit_list)))
    chinamutualfundbondportfolio_F_PRT_ENDDATE_list = pd.DataFrame(chinamutualfundbondportfolio_rating.F_PRT_ENDDATE.drop_duplicates().reset_index(drop = True))


    # Selection Criteria III-I: Split the credit rating preference
    chinamutualfundbondportfolio_rating_lower_credit_rating_selected = chinamutualfundbondportfolio_rating[chinamutualfundbondportfolio_rating.B_INFO_CREDITRATING.isin(chinamutualfundbondportfolio_rating_lower_credit_list)]
    chinamutualfundbondportfolio_rating_lower_credit_rating_selected_Assistant_ = pd.DataFrame()
    for i in range(len(chinamutualfundbondportfolio_F_PRT_ENDDATE_list)):
        chinamutualfundbondportfolio_rating_Assistant = chinamutualfundbondportfolio_rating_lower_credit_rating_selected[chinamutualfundbondportfolio_rating_lower_credit_rating_selected.F_PRT_ENDDATE == chinamutualfundbondportfolio_F_PRT_ENDDATE_list.iloc[i,:].values[0]].groupby(chinamutualfundbondportfolio_rating_lower_credit_rating_selected.S_INFO_WINDCODE_x).F_PRT_BDVALUETONAV.sum()
        chinamutualfundbondportfolio_rating_Assistant = pd.DataFrame(chinamutualfundbondportfolio_rating_Assistant)
        chinamutualfundbondportfolio_rating_Assistant.columns = [chinamutualfundbondportfolio_F_PRT_ENDDATE_list.iloc[i,:].values[0]]    
        chinamutualfundbondportfolio_rating_lower_credit_rating_selected_Assistant_ = pd.merge(chinamutualfundbondportfolio_rating_lower_credit_rating_selected_Assistant_, chinamutualfundbondportfolio_rating_Assistant, left_index=True, right_index=True, how='outer' )

    chinamutualfundbondportfolio_rating_high_credit_rating_selected = chinamutualfundbondportfolio_rating[chinamutualfundbondportfolio_rating.B_INFO_CREDITRATING.isin(chinamutualfundbondportfolio_rating_high_credit_list)]
    chinamutualfundbondportfolio_rating_high_credit_rating_selected_Assistant_ = pd.DataFrame()
    for i in range(len(chinamutualfundbondportfolio_F_PRT_ENDDATE_list)):
        chinamutualfundbondportfolio_rating_Assistant = chinamutualfundbondportfolio_rating_high_credit_rating_selected[chinamutualfundbondportfolio_rating_high_credit_rating_selected.F_PRT_ENDDATE == chinamutualfundbondportfolio_F_PRT_ENDDATE_list.iloc[i,:].values[0]].groupby(chinamutualfundbondportfolio_rating_high_credit_rating_selected.S_INFO_WINDCODE_x).F_PRT_BDVALUETONAV.sum()
        chinamutualfundbondportfolio_rating_Assistant = pd.DataFrame(chinamutualfundbondportfolio_rating_Assistant)
        chinamutualfundbondportfolio_rating_Assistant.columns = [chinamutualfundbondportfolio_F_PRT_ENDDATE_list.iloc[i,:].values[0]]    
        chinamutualfundbondportfolio_rating_high_credit_rating_selected_Assistant_ = pd.merge(chinamutualfundbondportfolio_rating_high_credit_rating_selected_Assistant_, chinamutualfundbondportfolio_rating_Assistant, left_index=True, right_index=True, how='outer' )

    chinamutualfundbondportfolio_rating_lower_credit_rating_selected_Assistant_ = chinamutualfundbondportfolio_rating_lower_credit_rating_selected_Assistant_.T
    chinamutualfundbondportfolio_rating_high_credit_rating_selected_Assistant_ = chinamutualfundbondportfolio_rating_high_credit_rating_selected_Assistant_.T

    # In[2] Countinue_ Rule based fund pool selection
    # Selection Criteria III-II: Split the credit rating preference
    Fund_high_credit_rating = chinamutualfundbondportfolio_rating_high_credit_rating_selected_Assistant_[(chinamutualfundbondportfolio_rating_high_credit_rating_selected_Assistant_ > 5) | (chinamutualfundbondportfolio_rating_high_credit_rating_selected_Assistant_ < 100)]
    Fund_lower_credit_rating = chinamutualfundbondportfolio_rating_lower_credit_rating_selected_Assistant_[(chinamutualfundbondportfolio_rating_lower_credit_rating_selected_Assistant_ > 15) | (chinamutualfundbondportfolio_rating_lower_credit_rating_selected_Assistant_ < 40)]

    Fund_high_low_credit_rating = Fund_high_credit_rating[list(set(Fund_high_credit_rating.columns).intersection(Fund_lower_credit_rating.columns))] * Fund_lower_credit_rating[list(set(Fund_high_credit_rating.columns).intersection(Fund_lower_credit_rating.columns))]
    Fund_high_low_credit_rating_pool = Fund_high_low_credit_rating[list(set(Mutual_Fund_Selected_Pool_.columns).intersection(Fund_high_low_credit_rating.columns))] * Mutual_Fund_Selected_Pool_[list(set(Mutual_Fund_Selected_Pool_.columns).intersection(Fund_high_low_credit_rating.columns))]
    Fund_high_low_credit_rating_pool = Fund_high_low_credit_rating_pool.applymap(lambda x: 1 if x > 0 else x)
    Fund_high_low_credit_rating_pool = Fund_high_low_credit_rating_pool.dropna(axis=1, thresh=1)
    

    Fund_high_low_credit_rating_pool.index = Fund_high_low_credit_rating_pool.index.astype(str)
    Fund_high_low_credit_rating_pool = Fund_high_low_credit_rating_pool[Fund_high_low_credit_rating_pool.index.map(pd.Timestamp) >= pd.Timestamp('20100101')]
    Fund_high_low_credit_rating_pool = Fund_high_low_credit_rating_pool.dropna(axis=0, thresh=1)

    Fund_high_low_credit_rating_pool_selected = Fund_high_low_credit_rating_pool.groupby(Fund_high_low_credit_rating_pool.index.map(pd.Timestamp).strftime('%Y-%m')).sum()
    # Only sets the periodic reports from mutual funds at Match, June, September and December as valided ones

    Fund_high_low_credit_rating_pool_selected = Fund_high_low_credit_rating_pool_selected[pd.Index(Fund_high_low_credit_rating_pool_selected.index.map(pd.Timestamp).strftime('%m')).isin([str('03'), str('06'), str('09'), str('12')])]
    # Adds the last calender days at the end of months described above
    Fund_high_low_credit_rating_pool_selected.index = [pd.Timestamp(Fund_high_low_credit_rating_pool_selected.index.map(pd.Timestamp).strftime('%Y')[i] + Fund_high_low_credit_rating_pool_selected.index.map(pd.Timestamp).strftime('%m')[i] + str(calendar.monthrange(year = int(Fund_high_low_credit_rating_pool_selected.index.map(pd.Timestamp).strftime('%Y')[i]), month = int(Fund_high_low_credit_rating_pool_selected.index.map(pd.Timestamp).strftime('%m')[i]))[1])) for i in range(Fund_high_low_credit_rating_pool_selected.shape[0])]
    # ra_fund_nva is a calender days recored net asset value datasets, here deletes the fund which initial established after the first day of 2018

    ra_fund_nav_ = ra_fund_nav.dropna(axis=1, thresh = (pd.Timestamp.today() - pd.Timestamp('20180101')).days )
    ra_fund_nav_.index = ra_fund_nav_.index.map(pd.Timestamp)
    ra_fund_nav_ = ra_fund_nav_[ra_fund_nav_.index.isin(tradedate_list)]
    #pd.DataFrame(ra_fund_nav_[set(ra_fund_nav_.columns).intersection(set(Fund_high_low_credit_rating_pool_selected.columns.map(lambda x: x.split('.')[0])))][ra_fund_nav_.index.map(pd.Timestamp) > pd.Timestamp('2012-01-01')].pct_change().add(1).cumprod().tail(1)).T.describe()

    Fund_high_low_credit_rating_pool_selected = Fund_high_low_credit_rating_pool_selected.reindex(ra_fund_nav_.index)
    Fund_high_low_credit_rating_pool_selected = Fund_high_low_credit_rating_pool_selected.ffill()
    Fund_high_low_credit_rating_pool_selected.columns = Fund_high_low_credit_rating_pool_selected.columns.map(lambda x: x.split('.')[0])
    Fund_high_low_credit_rating_pool_selected = Fund_high_low_credit_rating_pool_selected.loc[:,~Fund_high_low_credit_rating_pool_selected.columns.duplicated()]
    # Since the report would be published at the end of the next month of report period
    Fund_high_low_credit_rating_pool_selected = Fund_high_low_credit_rating_pool_selected.shift(periods = 22)

    ra_fund_nav_ = ra_fund_nav_[list(set(ra_fund_nav_.columns).intersection(set(Fund_high_low_credit_rating_pool_selected.columns)))]

    # In[3] Calculate momentum based result

    def Momentum_EW(stock_return,R,H):
    #    windows in month: Rolling Period as R Holding Period as H
        R=int(R*244//12)
        H=int(H*244/12)
        
        Momentum = stock_return.iloc[-R:,:].add(1).cumprod().tail(1)
    #    Equally weighted L-S strategy at 30% rank quantile of portfolio 
        Nb_stock = min(10, stock_return.shape[1])
        # Delete the top momentum result
        delete_top_number = min(2, max(stock_return.shape[1] - 6, 0) )
        
        Weight = Momentum.rank(ascending=False,axis=1).applymap(lambda x: 1/Nb_stock if ( x <= Nb_stock + delete_top_number) and (x > delete_top_number) else float('nan'))  
        return Weight



    start_date = pd.Timestamp('20120801')

    Regression_Period = 1
    Holding_Period = 1.2
    #ra_fund_nav_selected = ra_fund_nav_.copy()

    Weight_index_Momentum_Time_Series_ = pd.DataFrame()
    for i in range( (pd.Timestamp.today() - start_date - pd.Timedelta('1 days')).days // (int(365*Holding_Period)//12) + 1 ):
        
        selected_funded_date = '20120801'
        # adds one days here for the unexpected bug
        ra_fund_nav_selected = ra_fund_nav_[ra_fund_nav_.index <= pd.Timestamp(selected_funded_date) + pd.Timedelta( str( i * (int(365*Holding_Period)//12) + 3 ) + str(' days') )]
        
        # reset nav of the selected funds and set the funds initial period over a fixed dates that I set 1 calender year here: Criteria
        Fund_funding_criteria = '365 days'
        ra_fund_nav_selected = ra_fund_nav_selected[ ra_fund_nav_selected.index >= pd.Timestamp(selected_funded_date) + pd.Timedelta( str( i * (int(365*Holding_Period)//12) + 1 ) + str(' days') ) - pd.Timedelta(Fund_funding_criteria) ]
    #    ra_fund_nav_selected = ra_fund_nav_selected / ra_fund_nav_selected.iloc[0,:]

        ra_fund_nav_selected = ra_fund_nav_selected.dropna(axis=1, how='any')

        ra_fund_nav_selected_credit_exposure_criteria = Fund_high_low_credit_rating_pool_selected[Fund_high_low_credit_rating_pool_selected.index == ra_fund_nav_selected.index[0]]
        codes = list(set(ra_fund_nav_selected_credit_exposure_criteria.columns).intersection(set(ra_fund_nav_selected.columns)))
        ra_fund_nav_selected_credit_exposure_criteria = ra_fund_nav_selected_credit_exposure_criteria[codes]
        ra_fund_nav_selected = ra_fund_nav_selected[codes]
        
        ra_fund_nav_selected_ = ra_fund_nav_selected.multiply(ra_fund_nav_selected_credit_exposure_criteria.iloc[0], axis=1)
        ra_fund_nav_selected_ = ra_fund_nav_selected_.applymap(lambda x: np.nan if x == 0 else x)
        ra_fund_nav_selected_ = ra_fund_nav_selected_.dropna(axis=1, how='any')

            
        Weight_index_Momentum = Momentum_EW(ra_fund_nav_selected_.pct_change().dropna(), R=Regression_Period, H=Holding_Period)

    #    print(ra_fund_nav_selected.index[0], ra_fund_nav_selected.index[-1])
    #    print(Weight_index_Momentum.index)
    #   
        Weight_index_Momentum_Time_Series = Weight_index_Momentum.T
        Weight_index_Momentum_Time_Series_ = pd.merge(Weight_index_Momentum_Time_Series_, Weight_index_Momentum_Time_Series, left_index=True, right_index=True, how = 'outer')
    
    Weight_index_Momentum_Time_Series_ = Weight_index_Momentum_Time_Series_.T
    Weight_index_Momentum_Time_Series_.index = Weight_index_Momentum_Time_Series_.index.map(pd.Timestamp)
    Weight_index_Momentum_Time_Series_ = Weight_index_Momentum_Time_Series_.div(Weight_index_Momentum_Time_Series_.sum(axis=1), axis=0)

    Weight_index_Momentum_Time_Series_.sum(axis=1)



    #A = ra_fund_nav_selected.multiply(ra_fund_nav_selected_credit_exposure_criteria.values, axis=1)

    # In[3] Calculate momentum based weight with fill na ffill

    Fund_Weight = pd.DataFrame()
    for i in range(Weight_index_Momentum_Time_Series_.shape[0]):
        if i < Weight_index_Momentum_Time_Series_.shape[0]-1:
            ra_fund_nav_index = ra_fund_nav_[(ra_fund_nav_.index >= Weight_index_Momentum_Time_Series_.index[i]) & (ra_fund_nav_.index < Weight_index_Momentum_Time_Series_.index[i+1])].index
            Fund_Weight_ = Weight_index_Momentum_Time_Series_.reindex(ra_fund_nav_index)
            Fund_Weight_ = Fund_Weight_.ffill()
            Fund_Weight_ = Fund_Weight_.T
            Fund_Weight = pd.merge(Fund_Weight, Fund_Weight_, left_index=True, right_index=True, how = 'outer')
    #    elif i = Weight_index_Momentum_Time_Series_.shape[0]-1:
        else:
            ra_fund_nav_index = ra_fund_nav_[ra_fund_nav_.index >= Weight_index_Momentum_Time_Series_.index[i]].index
            Fund_Weight_ = Weight_index_Momentum_Time_Series_.reindex(ra_fund_nav_index)
            Fund_Weight_ = Fund_Weight_.ffill()
            Fund_Weight_ = Fund_Weight_.T
            Fund_Weight = pd.merge(Fund_Weight, Fund_Weight_, left_index=True, right_index=True, how = 'outer')

    Fund_Weight = Fund_Weight.T

    # In[3] Calculate backtesting result

    ra_fund_nav_ranged = ra_fund_nav_.pct_change()[ra_fund_nav_.index >= Fund_Weight.index[0]]
    Portfolio_Fund_Ret = Fund_Weight * ra_fund_nav_ranged
    Portfolio_Fund_Ret = Portfolio_Fund_Ret.sum(axis=1)
    AAAAA = Portfolio_Fund_Ret.add(1).cumprod()

    # In[3] DataBase

    ra_portfolio_pos = Weight_index_Momentum_Time_Series_.copy()
    ra_portfolio_pos = ra_portfolio_pos.stack().reset_index()
    ra_portfolio_pos.columns = ['ra_date', 'ra_code', 'ra_fund_ratio']
    ra_portfolio_pos = pd.merge(ra_portfolio_pos, ra_fund[['globalid', 'ra_code']], left_on=['ra_code'], right_on=['ra_code'], how = 'inner')

    ra_portfolio_pos.columns = ['ra_date', 'ra_code', 'ra_position', 'globalid']

    ra_portfolio_pos = ra_portfolio_pos[ra_portfolio_pos.ra_position!=0]

    np.sum(ra_portfolio_pos.pivot_table('ra_position', 'ra_date', 'globalid').sum(axis=1))

    ra_portfolio_pos = ra_portfolio_pos.applymap(str)

    #ra_portfolio_pos = ra_portfolio_pos[ra_portfolio_pos.ra_position.map(np.float64) > 0.001]
    np.sort(ra_portfolio_pos.ra_date.drop_duplicates())

     

    # In[1]: Condition in Credit rating Write data to database

    print(ra_portfolio_pos)

    conn = pymysql.connect(host='192.168.88.254', user='root', passwd='Mofang123', database='asset_allocation', charset='utf8')
    x = conn.cursor()
    for i in range(ra_portfolio_pos.shape[0]):
        
        try:
           excuse_ = "INSERT INTO `ra_portfolio_pos` (`ra_portfolio_id`, `ra_date`, `ra_pool_id`, `ra_fund_id`, `ra_fund_code`, `ra_fund_type`, `ra_fund_ratio`, `created_at`, `updated_at`) VALUES ('PO.TEST30', '%s', '1111010111', '%s','%s', '11101',' %s', '2019-04-30 17:04:39', '2019-04-30 17:04:39');" %  (ra_portfolio_pos.iloc[i,0], ra_portfolio_pos.iloc[i,3], ra_portfolio_pos.iloc[i,1], ra_portfolio_pos.iloc[i,2])
        #   print(excuse_)
        
           x.execute(excuse_)
           conn.commit()
        except:
           conn.rollback()

    conn.close()
