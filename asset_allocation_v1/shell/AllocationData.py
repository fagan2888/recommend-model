#coding=utf8



import os
import MySQLdb
from datetime import datetime
import DBData
from Const import datapath


db_params = {
            "host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
            "port": 3306,
            "user": "koudai",
            "passwd": "Mofang123",
            "db":"mofang_api",
            "charset": "utf8"
        }
db_params = {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "passwd": "Mofang123",
            "db":"mofang",
            "charset": "utf8"
        }




class allocationdata:


    fund_id_code_dict = {}
    fund_code_id_dict = {}


    def __init__(self):

        conn  = MySQLdb.connect(**db_params)

        cursor = conn.cursor()
        sql = "select fi_globalid, fi_code from fund_infos"

        cursor.execute(sql)
        records = cursor.fetchall()
        for record in records:
            self.fund_id_code_dict[record[0]] = record[1]
            self.fund_code_id_dict[record[1]] = record[0]

        sql = "select ii_globalid, ii_index_code from index_info"

        cursor.execute(sql)
        records = cursor.fetchall()
        for record in records:
            self.fund_id_code_dict[record[0]] = record[1]
            self.fund_code_id_dict[record[1]] = record[0]


        conn.commit()
        conn.close()



    start_date                              = '2010-01-01'

    end_date                                = datetime.now().strftime('%Y-%m-%d')
    fund_measure_lookback                   = 52              #回溯52个周
    fund_measure_adjust_period              = 26              #26个周重新算基金池


    jensen_ratio                            = 0.5             #jensen取前50%
    sortino_ratio                           = 0.5
    ppw_ratio                               = 0.5
    stability                               = 0.5


    fixed_risk_asset_risk_lookback          = 52
    fixed_risk_asset_risk_adjust_period     = 5


    allocation_lookback                     = 13
    allocation_adjust_period                = 13



    stock_fund_measure = {}
    stock_fund_label   = {}
    bond_fund_measure  = {}
    bond_fund_label    = {}
    money_fund_measure = {}
    money_fund_label   = {}
    other_fund_measure = {}
    other_fund_label   = {}


    label_asset_df = None
    stock_fund_df  = None
    bond_fund_df   = None
    money_fund_sharpe_df  = None
    other_fund_sharpe_df  = None

    equal_risk_asset_ratio_df = None
    equal_risk_asset_df       = None


    high_risk_position_df    = None
    low_risk_position_df     = None
    highlow_risk_position_df = None


    high_risk_asset_df       = None
    low_risk_asset_df        = None
    highlow_risk_asset_df    = None

    riskhighlowriskasset     = None

    #stock_df = None
    #bond_df  = None
    #money_df = None
    #other_df = None
    #index_df = None
    data_start_date = None
    #position_df = None
    #scale_df    = None


    def all_dates(self):

        dates = DBData.trade_dates('1900-01-01',self.start_date)
        start_n = len(dates) - self.fund_measure_lookback    - self.fixed_risk_asset_risk_lookback  - 2 * self.allocation_lookback
        self.data_start_date = dates[len(dates) - self.fund_measure_lookback    - self.fixed_risk_asset_risk_lookback  - 2 * self.allocation_lookback]
        dates = dates[start_n  : len(dates)]

        print self.data_start_date

        #self.stock_df    = DBData.stock_fund_value(self.data_start_date, self.end_date)
        #print 'load stock done'
        #self.bond_df     = DBData.bond_fund_value(self.data_start_date, self.end_date)
        #print 'load bond done'
        #self.money_df    = DBData.money_fund_value(self.data_start_date, self.end_date)
        #print 'load money done'
        #self.other_df    = DBData.other_fund_value(self.data_start_date, self.end_date)
        #print 'load other done'
        #self.index_df    = DBData.index_value(self.data_start_date, self.end_date)
        #print 'load index done'
        #self.position_df = DBData.position()
        #print 'position done'
        #self.scale_df    = DBData.scale()
        #print 'scale done'
        #print 'load data done'

        #self.stock_df.to_csv(datapath('stock.csv'))
        #self.bond_df.to_csv(datapath('bond.csv'))
        #self.money_df.to_csv(datapath('money.csv'))
        #self.other_df.to_csv(datapath('other.csv'))
        #self.index_df.to_csv(datapath('index.csv'))
        #self.position_df.to_csv(datapath('position.csv'))
        #self.scale_df.to_csv(datapath('scale.csv'))

        return 0


    def get_date_df(df, start_date, end_date):
        _df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
        _df = _df[ _df.index >= datetime.strptime(start_date,'%Y-%m-%d')]
        return _df
