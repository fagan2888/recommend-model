#coding=utf8


import MySQLdb
from datetime import datetime


class allocationdata:


	fund_id_code_dict = {}
	fund_code_id_dict = {}


	def __init__(self):

		conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='mofang', charset='utf8')

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


	equal_risk_asset_ratio_df = None
	equal_risk_asset_df       = None


	high_risk_position_df    = None
	low_risk_position_df     = None
	highlow_risk_position_df = None


	high_risk_asset_df       = None
	low_risk_asset_df        = None
	highlow_risk_asset_df    = None



