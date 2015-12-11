#coding=utf8

import string
import MySQLdb
import datetime
import numpy as np
from math import e

today = datetime.datetime.now().strftime('%Y-%m-%d')


#易方达创业板ETF联接 易方达沪深300ETF联接 南方中证500ETF联接(LOF) 易方达稳健收益债券A 南方通利债券A 建信稳定增利债券C 华安黄金易(ETF联接)A  嘉实货币A
funds = [110026, 110020, 160119, 110007, 563, 530008, 216, 70008]
fids  = [30000714, 30000793, 30000621, 30001895, 30002145 , 30001876, 30000696, 30003314]


def normalized(ws):
        s = 0.0
        weights = []
        for w in ws:
                s = s + w
        for w in ws:
                weights.append( 1.0 * w / s )
        return weights


risk_weights      =     {}

risk_weights[0]   =     [0, 0, 0, 0, 0, 0, 0, 1]
risk_weights[1]   =     [0.0071, 0.0107, 0.0178, 0.3410, 0.3220, 0.3015, 0, 0]
risk_weights[2]   =     [0.0213, 0.0320, 0.0534, 0.2886, 0.2970, 0.3078, 0, 0]
risk_weights[3]   =     [0.0315, 0.0473, 0.0788, 0.2700, 0.2816, 0.2908, 0, 0]
risk_weights[4]   =     [0.0416, 0.0625, 0.1041, 0.2330, 0.2939, 0.2648, 0, 0]
risk_weights[5]   =     [0.0518, 0.0777, 0.1295, 0.2279, 0.2461, 0.2670, 0, 0]
risk_weights[6]   =     [0.0619, 0.0929, 0.1548, 0.2106, 0.2296, 0.2501, 0, 0]
risk_weights[7]   =     [0.0802, 0.1203, 0.2006, 0.2096, 0.1996, 0.1896, 0, 0]
risk_weights[8]   =     [0.1005, 0.1508, 0.2513, 0.2087, 0.2887, 0, 0, 0]
risk_weights[9]   =     [0.1208, 0.1812, 0.3020, 0.2380, 0.1578, 0, 0, 0]
risk_weights[10]  =     [0.1411, 0.2117, 0.3528, 0.1272, 0.1672, 0, 0, 0]


portfolio_base_sql = "replace into portfolios (p_name, p_risk, pf_date, pf_position_reason,pf_focus, pf_position_record, pf_annual_returns, pf_expect_returns_max, pf_expect_returns_min, created_at, updated_at) values ('%s', '%f', '%s','%s' ,'%s', '%s', '%f', '%f', '%f', '%s', '%s')"

select_pid_base_sql = "select id from portfolios where p_name = '%s'"


weight_base_sql = "replace into portfolio_weights (pw_portfolio_id, pw_portfolio_name, pw_fund_id, pw_weight, pw_risk, created_at, updated_at) values ('%d', '%s', '%d', '%f', '%f', '%s', '%s')"



conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='recommend', charset='utf8')


for i in range(0, 11):
	name = 'risk' + str(i) +'_' + today
	sql = portfolio_base_sql % (name, 1.0 * i / 10, today,'reason','focus', '固定周期调仓，优化资产配置比例。', 0.00, 0.00 , 0.00, datetime.datetime.now(), datetime.datetime.now())
	cur = conn.cursor()
	cur.execute(sql)
	cur.close()
	conn.commit()

	sql = select_pid_base_sql % (name)
	cur = conn.cursor()
	cur.execute(sql)
	record = cur.fetchone()
	pid = record[0]
	cur.close()
	conn.commit()

	weight = normalized(risk_weights[i])
	cur = conn.cursor()
        for n in range(0, len(weight)):
		sql = weight_base_sql % (pid, name, fids[n], weight[n], 1.0 * i / 10, datetime.datetime.now(), datetime.datetime.now())
                cur.execute(sql)
        cur.close()
        conn.commit()				




update_copywrite_base_sql = "update portfolios set pf_position_reason = '%s' where pf_date = '%s' and p_risk = '%f'"
cur = conn.cursor()
sql = update_copywrite_base_sql % ('全仓配置货币基金，享受安全稳健收益。', today, 0.0)
cur.execute(sql)
for i in range(1,11):
        sql = update_copywrite_base_sql % ('根据市场行情对配置方案进行调整，平衡收益和风险。', today, 1.0 * i / 10)
        cur.execute(sql)

cur.close()
conn.commit()


update_copywrite_base_sql = "update portfolios set pf_position_record = '%s' where pf_date = '%s' and p_risk = '%f'"
cur = conn.cursor()
for i in range(0, 11):
        sql = update_copywrite_base_sql % ('固定周期调仓，优化资产配置比例。', today, 1.0 * i / 10)
        cur.execute(sql)
cur.close()
conn.commit()


update_copywrite_base_sql = "update portfolios set pf_focus = '%s' where pf_date = '%s' and p_risk = '%f'"
cur = conn.cursor()
for i in range(1, 11):
        sql = update_copywrite_base_sql % ('A股市场持续震荡，逐渐筑底成功，市场信心逐步恢复市场人气逐步回暖。', today, 1.0 * i / 10)
        cur.execute(sql)

sql = update_copywrite_base_sql % ('全仓配置货币基金，享受安全稳健收益。', today, 0)
cur.execute(sql)

cur.close()
conn.commit()



fund_value_sql = "select pv_risk, pv_date, pv_value from portfolio_values order by pv_date asc"
cur = conn.cursor()
values = {}
ret = cur.execute(fund_value_sql)
for record in cur.fetchall():
	vs = values.setdefault(record[0],[])	
	vs.append((float)(record[2]))	

cur.close()
conn.commit()

final_values = values

cur = conn.cursor()
update_return_base_sql = "update portfolios set  pf_annual_returns = '%f', pf_expect_returns_max = '%f', pf_expect_returns_min = '%f' where pf_date = '%s' and p_risk = '%f'"


for key in final_values.keys():

        values = final_values[key]
        length = len(values)
        p = values[-1] / values[0]
        annual_return = p ** (1.0 / (1.0 * length / 250)) - 1
        profits = []
        for i in range(1, len(values)):
                profits.append(values[i] / values[i - 1] - 1)
        sigma = np.std(profits)
        u = np.mean(profits)

        portfolio_p = e ** (u * 250)
        portfolio_upper_p = portfolio_p * ( 1 + sigma * (250 ** 0.5))
        annual_return_max = portfolio_p * ( 1 + sigma * (250 ** 0.5)) - 1
        annual_return_min = portfolio_p * ( 1 - sigma * (250 ** 0.5)) - 1
        sql = update_return_base_sql % (annual_return, annual_return_max, annual_return_min, today, 1.0 * key)
        cur.execute(sql)


conn.commit()
cur.close()

prediction_month_return = [0.0417, 0.0046, 0.0051, 0.0064, 0.0078, 0.0092, 0.0104, 0.0122, 0.0135, 0.0154, 0.0168]
prediction_annual_return = [0.0500, 0.0557, 0.0616, 0.0763, 0.0936, 0.1100, 0.1242, 0.1464, 0.1623, 0.1844, 0.2019]

cur = conn.cursor()
update_prediction_return_sql = "update portfolios set pf_prediction_month_returns = '%f', pf_prediction_annual_returns = '%f' where pf_date = '%s' and p_risk = '%f'"
for i in range(0, 11):
	sql = update_prediction_return_sql % (prediction_month_return[i], prediction_annual_return[i], today, 1.0 * i / 10)
	cur.execute(sql)

cur.close()
conn.commit()


conn.close()
