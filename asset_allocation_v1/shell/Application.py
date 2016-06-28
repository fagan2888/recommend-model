#coding=utf8



import string
import json
import sys
sys.path.append('shell')
import pandas as pd
import LabelAsset
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset
import os
import DB
import thread 
import MySQLdb
from datetime import datetime
import AllocationData
from flask import Flask
from flask import request
import time


app = Flask(__name__)


def asset_allocation(allocationdata, uid):

	#try:
	allocationdata.all_dates()
	LabelAsset.labelasset(allocationdata)
	EqualRiskAssetRatio.equalriskassetratio(allocationdata)
	EqualRiskAsset.equalriskasset(allocationdata)
	HighLowRiskAsset.highlowriskasset(allocationdata)
	DB.fund_measure(allocationdata)
	DB.label_asset(allocationdata)
	DB.asset_allocation(allocationdata)

	conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
	cursor = conn.cursor()

	sql = "update user_job_status set ujs_status = '%s'  where ujs_uid = %d"  % ('complete', uid)

	cursor.execute(sql)
	conn.commit()
	conn.close()

	#except:

	#	conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
	#	cursor = conn.cursor()

	#	sql = "update user_job_status set ujs_status = '%s'  where ujs_uid = %d"  % ('error', uid)
	#	cursor.execute(sql)
	#	conn.commit()
	#	conn.close()


	#return 0


#资产配置接口
@app.route("/asset_allocation_v1", methods=['GET', 'POST'])
def asset_allocation_v1():


	allocationdata = AllocationData.allocationdata()

	#print allocationdata.fund_code_id_dict

	args = request.form

	allocationdata.start_date                            = args.get('start_date')
	#allocationdata.start_date                            = '2015-12-01'
	allocationdata.fund_measure_lookback                 = string.atoi(args.get('fund_measure_lookback'))
	allocationdata.fund_measure_adjust_period            = string.atoi(args.get('fund_measure_adjust_period'))		
	allocationdata.jensen_ratio                          = string.atof(args.get('jensen_ratio'))		
	allocationdata.sortino_ratio                         = string.atof(args.get('sortino_ratio'))		
	allocationdata.ppw_ratio                             = string.atof(args.get('ppw_ratio'))		
	allocationdata.stability_ratio                       = string.atof(args.get('stability_ratio'))	
	allocationdata.fixed_risk_asset_lookback             = string.atoi(args.get('fixed_risk_asset_risk_lookback'))	
	allocationdata.fixed_risk_asset_risk_adjust_period   = string.atoi(args.get('fixed_risk_asset_risk_adjust_period'))	
	allocationdata.allocation_lookback                   = string.atoi(args.get('allocation_lookback'))
	allocationdata.allocation_adjust_period              = string.atoi(args.get('allocation_adjust_period'))	
	uid                                   = string.atoi(args.get('uid'))


	json_args                             = json.dumps(args)

	print json_args

	conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
	cursor = conn.cursor()
	sql = "insert into user_job_status (ujs_uid, ujs_args, ujs_status, created_at, updated_at) values(%d, '%s', '%s', '%s', '%s')" % (uid,json_args, 'running', datetime.now() ,datetime.now())
	cursor.execute(sql)
	conn.commit()
	conn.close()


	thread.start_new_thread(asset_allocation,(allocationdata, uid) )
	#asset_allocation(start_date)


	result = {}
	result['status'] = 'running'
	result['uid']     = uid


	ret            = {}
	ret['code']    = 20000
	ret['message'] = 'Success'
	ret['result']  = result

	
	return json.dumps(ret)



@app.route("/risk_asset_allocation_v1")
def risk_asset_allocation():

	args           =   request.args
	start_date     =   args.get('start_date')
	risk           =   string.atof(args.get('risk'))

	conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
	cursor = conn.cursor()


	ratio = {}
	ratio['2016-03-01'] = [{'fund':30000001, 'ratio':0.3}, {'fund':30000001, 'ratio':0.2}]
	nav                 = {'2016-03-01':1.0, '2016-03-02':1.2, '2016-03-03':1.1, '2016-03-04':1.15}


	result = {}
	result['ratio'] = ratio
	result['nav']   = nav

	
	#sql = "insert into risk_asset_allocation (raa_start_date, raa_risk, raa_result, raa_status, created_at, updated_at) values('%s', '%f', '%s', '%s', '%s', '%s')" % (start_date, risk,  json.dumps(result) ,'complete', datetime.now() ,datetime.now())
	#print sql
	#cursor.execute(sql)


	conn.commit()
	conn.close()

	result = {}
	result['status'] = 'running'

	ret            = {}
	ret['code']    = 20000
	ret['message'] = 'Success'
	ret['result']  = result


	return json.dumps(ret)


@app.route("/")
def hello():
	return 'Hello World'


if __name__ == "__main__":
	app.run()
