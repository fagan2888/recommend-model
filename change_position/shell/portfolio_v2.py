#coding=utf8

import string
import datetime
import time
import MySQLdb
import numpy as np
from math import e

dates = []

risks = {}
for i in range(1,11):
    risks.setdefault(i,[])

f = open('./data/change_position.csv','r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    vec = line.split(',')
    date_str = vec[0].strip()
    d = datetime.datetime.strptime(date_str, '%Y/%m/%d')
    date_str = d.strftime('%Y-%m-%d')
    dates.append(date_str)
    risks[1].append(vec[1:4])            
    risks[2].append(vec[4:7])            
    risks[3].append(vec[7:10])            
    risks[4].append(vec[10:13])            
    risks[5].append(vec[13:16])            
    risks[6].append(vec[16:19])            
    risks[7].append(vec[19:22])            
    risks[8].append(vec[22:25])            
    risks[9].append(vec[25:28])            
    risks[10].append(vec[28:31])    


def normalized(ws):
    s = 0.0
    weights = []
    for w in ws:
        s = s + w    
    for w in ws:
        weights.append( 1.0 * w / s )
    return weights        
    

#print dates

def date_str_list(start_date_str , end_date_str):
    date_list = []
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date   = datetime.datetime.strptime(end_date_str, '%Y-%m-%d') 
    delta = datetime.timedelta(days=1)
    d = start_date
    date_list.append(d.strftime('%Y-%m-%d'))
    while(d <= end_date):
        d = d + delta
        date_list.append(d.strftime('%Y-%m-%d')) 
    
    return date_list


def funds_values(funds, start_date, end_date):
    
    fids = []
    conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='mofang', charset='utf8')
    cur = conn.cursor()
    for fid in funds:
    #print fid
        cur.execute("select fi_globalid, fi_name from fund_infos where fi_code = '%d'" % fid)
        record = cur.fetchone()
        fids.append(record[0])
    
    cur.close()
    conn.commit()
    
    #print fids 
    f_values = {}
    for fid in fids:
        cur = conn.cursor()
        ret = cur.execute("select fv_time, fv_net_value, fv_authority_value from fund_value where fv_fund_id = %d and fv_time >= '%s' and fv_time <= '%s'" % (fid, start_date, end_date))   
    
        values = {}
        for record in cur.fetchall():

            date_str = str(record[0])
            net_value = string.atof(record[1])
            authority_value = string.atof(record[2])

            if authority_value == 0:
                continue
            values[date_str] = authority_value
            
        f_values[fid] = values
        
        cur.close()
        conn.commit()
        
    conn.close()
    
    
    date_list =  date_str_list(start_date, end_date)

    full_values = {}
    
    for i in range(1, len(date_list)):
        
        date_str = date_list[i]
        
        has_value = True
        date_vs = []
        for fid in fids:
            vs = f_values[fid]
            if not vs.has_key(date_str):
                has_value = False
                break
            else:
                date_vs.append(vs[date_str])
        
        if has_value == True:
            full_values[date_str] = date_vs
        
    return full_values


funds = [110026, 110020, 160119, 110007, 563, 530008, 216, 70008]
fids = [30000714, 30000793, 30000621, 30001895, 30002145 , 30001876, 30000696, 30003314]


#start_date = '2013-01-04'
start_date = dates[0]
now_time = datetime.datetime.now()
yes_time = now_time + datetime.timedelta(days=-1)
end_date   = yes_time.strftime('%Y-%m-%d')
#end_date   =  '2015-11-20'
full_values = funds_values(funds, start_date, end_date)
#print full_values


keys = full_values.keys()
keys.sort()

final_values = {}

for i in range(1, 11):

    risk = risks[i]
    weight = [0, 0, 0, 0, 0, 0, 0]
    r_w = risk[0]
    weight[0] =    string.atof(r_w[1].strip()) * 0.2 
    weight[1] =    string.atof(r_w[1].strip()) * 0.3
    weight[2] =    string.atof(r_w[1].strip()) * 0.5 

    if(string.atof(r_w[2].strip()) <= 0.5):
        weight[3] = string.atof(r_w[2].strip()) * 0.5        
        weight[4] = string.atof(r_w[2].strip()) * 0.5        
        weight[5] = 0
    elif(string.atof(r_w[2].strip()) >= 0.51 and string.atof(r_w[2].strip()) <= 1):
        weight[3] = string.atof(r_w[2].strip()) * 1.0 / 3.0
        weight[4] = string.atof(r_w[2].strip()) * 1.0 / 3.0    
        weight[5] = string.atof(r_w[2].strip()) * 1.0 / 3.0        

    weight[6] =    string.atof(r_w[0].strip()) * 1.0 
    weight    =        normalized(weight)

    values = []    
    values.append(1)

    for j in range(1, len(keys)):

        key = keys[j]
        try:
            if dates.index(key) >= 0:
                index = dates.index(key)
                r_w = risk[index]
                weight[0] =    string.atof(r_w[1].strip()) * 0.2 
                weight[1] =    string.atof(r_w[1].strip()) * 0.3
                weight[2] =    string.atof(r_w[1].strip()) * 0.5 

                if(string.atof(r_w[2].strip()) <= 0.5):
                    weight[3] = string.atof(r_w[2].strip()) * 0.5        
                    weight[4] = string.atof(r_w[2].strip()) * 0.5        
                    weight[5] = 0
                elif(string.atof(r_w[2].strip()) >= 0.51 and string.atof(r_w[2].strip()) <= 1):
                    weight[3] = string.atof(r_w[2].strip()) * 1.0 / 3.0
                    weight[4] = string.atof(r_w[2].strip()) * 1.0 / 3.0    
                    weight[5] = string.atof(r_w[2].strip()) * 1.0 / 3.0        

                weight[6] =    string.atof(r_w[0].strip()) * 1.0 
                weight    =        normalized(weight)

        except:
            pass


        today           = keys[j]
        yesterday     = keys[j-1]
        
        today_value     = full_values[today]        
        yesterday_value = full_values[yesterday]
        
        p = 0
        for n in range(0, len(today_value) - 1):
            p = p + (today_value[n] / yesterday_value[n] - 1) * weight[n]        
        v = values[len(values) -1]
        values.append(v * (1 + p))    
         
    final_values[i] = values    

    #print risk

#for key in final_values.keys():
#    print key
#    print risks[key]
#    print final_values[key]


conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='recommend', charset='utf8')


portfolio_base_sql = "replace into portfolios (p_name, p_risk, pf_date, pf_position_reason,pf_focus, pf_position_record, pf_annual_returns, pf_expect_returns_max, pf_expect_returns_min, created_at, updated_at) values ('%s', '%f', '%s','%s' ,'%s', '%s', '%f', '%f', '%f', '%s', '%s')"


select_pid_base_sql = "select id from portfolios where p_name = '%s'"


weight_base_sql = "replace into portfolio_weights (pw_portfolio_id, pw_portfolio_name, pw_fund_id, pw_weight, pw_risk, created_at, updated_at) values ('%d', '%s', '%d', '%f', '%f', '%s', '%s')"


for i in range(0, 11):

    risk = []    
    weight = [0, 0, 0, 0, 0, 0, 0, 0]

    if i > 0:

        risk      =     risks[i]
        r_w        =     risk[0]    
        weight[0] =    string.atof(r_w[1].strip()) * 0.2 
        weight[1] =    string.atof(r_w[1].strip()) * 0.3
        weight[2] =    string.atof(r_w[1].strip()) * 0.5 

        if(string.atof(r_w[2].strip()) <= 0.5):
            weight[3] = string.atof(r_w[2].strip()) * 0.5        
            weight[4] = string.atof(r_w[2].strip()) * 0.5        
            weight[5] = 0
        elif(string.atof(r_w[2].strip()) >= 0.51 and string.atof(r_w[2].strip()) <= 1):
            weight[3] = string.atof(r_w[2].strip()) * 1.0 / 3.0
            weight[4] = string.atof(r_w[2].strip()) * 1.0 / 3.0    
            weight[5] = string.atof(r_w[2].strip()) * 1.0 / 3.0        

        weight[6] =    string.atof(r_w[0].strip()) * 1.0 
        weight    =        normalized(weight)


    for j in range(0, len(dates)):

        date_time = datetime.datetime.strptime(dates[j],'%Y-%m-%d')
        date_time_threshold = datetime.datetime.strptime('2014-10-01','%Y-%m-%d')                     
        if(date_time < date_time_threshold):
            continue                
        date_str = datetime.datetime.strptime(dates[j],'%Y-%m-%d').strftime('%Y%m%d')
        name = 'risk' + str(i) +'_' + dates[j] 
        sql = portfolio_base_sql % (name, 1.0 * i / 10, dates[j],'reason','focus', '固定周期调仓，优化资产配置比例。', 0.07, 0.10,0.05, datetime.datetime.now(), datetime.datetime.now())
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
    
        if i > 0:

            r_w = risk[j]
            weight = [0, 0, 0, 0, 0, 0, 0, 0]
            weight[0] =    string.atof(r_w[1].strip()) * 0.2 
            weight[1] =    string.atof(r_w[1].strip()) * 0.3
            weight[2] =    string.atof(r_w[1].strip()) * 0.5 

            if(string.atof(r_w[2].strip()) <= 0.5):
                weight[3] = string.atof(r_w[2].strip()) * 0.5        
                weight[4] = string.atof(r_w[2].strip()) * 0.5        
                weight[5] = 0
            elif(string.atof(r_w[2].strip()) >= 0.51 and string.atof(r_w[2].strip()) <= 1):
                weight[3] = string.atof(r_w[2].strip()) * 1.0 / 3.0
                weight[4] = string.atof(r_w[2].strip()) * 1.0 / 3.0    
                weight[5] = string.atof(r_w[2].strip()) * 1.0 / 3.0        

            weight[6] =    string.atof(r_w[0].strip()) * 1.0 
            weight    =     normalized(weight)

        else:
            weight = [0,0,0,0,0,0,0,1]

        cur = conn.cursor()    
        for n in range(0, len(weight)):
            sql = weight_base_sql % (pid, name, fids[n], weight[n], 1.0 * i / 10, datetime.datetime.now(), datetime.datetime.now())    
            cur.execute(sql)
        cur.close()
        conn.commit()                    


keys = full_values.keys()
keys.sort()

values_base_sql = "replace into portfolio_values (pv_risk, pv_date, pv_value, pv_ratio, created_at, updated_at) values ('%f', '%s', '%f', '%f', '%s', '%s')"

cur = conn.cursor()
for i in range(1, 11):
    values = final_values[i]
    for j in range(0 ,len(keys)):
        ratio = 0.0
        if j > 0:
            ratio = values[j] / values[j-1] - 1         
        sql = values_base_sql % (1.0 * i / 10, keys[j], values[j], ratio, datetime.datetime.now(), datetime.datetime.now())    
        cur.execute(sql)    

cur.close()
conn.commit()


cur = conn.cursor()
for j in range(0, len(keys)):

    today_vs = full_values[keys[j]]
    today_v  = today_vs[len(today_vs) - 1]    

    ratio = 0.0
    if j > 0:

        yestoday_vs = full_values[keys[j - 1]]
        yestoday_v  = today_vs[len(yestoday_vs) - 1]    
        ratio = today_v / yestoday_v - 1         
    sql = values_base_sql % (0.0 * i / 10, keys[j], today_v, ratio, datetime.datetime.now(), datetime.datetime.now())    
    cur.execute(sql)    
cur.close()
conn.commit()


update_copywrite_base_sql = "update portfolios set pf_position_reason = '%s' where pf_date = '%s' and p_risk = '%f'"
cur = conn.cursor()
sql = update_copywrite_base_sql % ('全仓配置货币基金，享受安全稳健收益。', '2015-09-11', 0.0)
cur.execute(sql)
for i in range(1,11):
    sql = update_copywrite_base_sql % ('根据市场行情对配置方案进行调整，平衡收益和风险。', '2015-09-11', 1.0 * i / 10)
    cur.execute(sql)

cur.close()
conn.commit()

update_copywrite_base_sql = "update portfolios set pf_position_record = '%s' where pf_date = '%s' and p_risk = '%f'"
cur = conn.cursor()
for i in range(0, 11):
    sql = update_copywrite_base_sql % ('固定周期调仓，优化资产配置比例。', '2015-09-11', 1.0 * i / 10)
    cur.execute(sql)

for i in range(0, 11):
    sql = update_copywrite_base_sql % ('固定周期调仓，优化资产配置比例。', '2015-06-19', 1.0 * i / 10)
    cur.execute(sql)

for i in range(0, 11):
    sql = update_copywrite_base_sql % ('固定周期调仓，优化资产配置比例。', '2015-03-27', 1.0 * i / 10)
    cur.execute(sql)

for i in range(0, 11):
    sql = update_copywrite_base_sql % ('固定周期调仓，优化资产配置比例。', '2014-12-31', 1.0 * i / 10)
    cur.execute(sql)
for i in range(0, 11):
    sql = update_copywrite_base_sql % ('固定周期调仓，优化资产配置比例。', '2014-10-10', 1.0 * i / 10)
    cur.execute(sql)
cur.close()
conn.commit()



update_copywrite_base_sql = "update portfolios set pf_focus = '%s' where pf_date = '%s' and p_risk = '%f'"
cur = conn.cursor()
for i in range(1, 11):
    sql = update_copywrite_base_sql % ('A股市场持续震荡，逐渐筑底成功，市场信心逐步恢复市场人气逐步回暖。', '2015-09-11', 1.0 * i / 10)
    cur.execute(sql)

for i in range(1, 11):
    sql = update_copywrite_base_sql % ('A股遭遇暴跌，市场信心遭受打击，市场流动性和资金面遭到严重考验，短期市场风险加大波动加剧。', '2015-06-19', 1.0 * i / 10)
    cur.execute(sql)

for i in range(1, 11):
    sql = update_copywrite_base_sql % ('A股牛市行情继续，国内货币政策持续宽松，宏观经济形势恶化，短期或风险加大波动加剧。', '2015-03-27', 1.0 * i / 10)
    cur.execute(sql)

for i in range(1, 11):
    sql = update_copywrite_base_sql % ('A股牛市曙光初现，债券市场继续走强。市场资金面持续宽松，风险偏好逐渐加强，板块轮动效应明显。', '2014-12-31', 1.0 * i / 10)
    cur.execute(sql)
for i in range(1, 11):
    sql = update_copywrite_base_sql % ('京津冀污染持续升级，环保产业扶持力度将加大，持续关注板块和个股的机会。大盘逐渐走强，或出现短期结构性行情。', '2014-10-10', 1.0 * i / 10)
    cur.execute(sql)
sql = update_copywrite_base_sql % ('全仓配置货币基金，享受安全稳健收益。', '2015-09-11', 0)
cur.execute(sql)
sql = update_copywrite_base_sql % ('全仓配置货币基金，享受安全稳健收益。', '2015-06-19', 0)
cur.execute(sql)
sql = update_copywrite_base_sql % ('全仓配置货币基金，享受安全稳健收益。', '2015-03-27', 0)
cur.execute(sql)
sql = update_copywrite_base_sql % ('全仓配置货币基金，享受安全稳健收益。', '2014-12-31', 0)
cur.execute(sql)
sql = update_copywrite_base_sql % ('全仓配置货币基金，享受安全稳健收益。', '2014-10-10', 0)
cur.execute(sql)
cur.close()
conn.commit()



cur = conn.cursor()
update_return_base_sql = "update portfolios set     pf_annual_returns = '%f', pf_expect_returns_max = '%f', pf_expect_returns_min = '%f' where pf_date = '%s' and p_risk = '%f'"
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
    annual_return_max = portfolio_p * ( 1 + sigma * (250 ** 0.5)) -    1
    annual_return_min = portfolio_p * ( 1 - sigma * (250 ** 0.5)) -    1
    sql = update_return_base_sql % (annual_return, annual_return_max, annual_return_min, '2015-09-11', 1.0 * key / 10)
    cur.execute(sql)



money_values = []
keys = full_values.keys()
keys.sort()
for key in keys:
    vs = full_values[key]
    money_values.append(vs[len(vs) - 1])     

length = len(money_values)
p = money_values[-1] / money_values[0]
annual_return = p ** (1.0 / (1.0 * length / 250)) - 1
profits = []    
for i in range(1, len(money_values)):
    profits.append(money_values[i] / money_values[i - 1] - 1)     
sigma = np.std(profits)
u = np.mean(profits)


portfolio_p = e ** (u * 250)
portfolio_upper_p = portfolio_p * ( 1 + sigma * (250 ** 0.5)) 
annual_return_max = portfolio_p * ( 1 + sigma * (250 ** 0.5)) -    1
annual_return_min = portfolio_p * ( 1 - sigma * (250 ** 0.5)) -    1
sql = update_return_base_sql % (annual_return, annual_return_max, annual_return_min, '2015-09-11', 0.0)
cur.execute(sql)

cur.close()
conn.commit()


cur = conn.cursor()
for key in final_values.keys():
    values = final_values[key]
    length = len(values)
    month_num = 1.0 * length / 20    
    month_returns = (values[length-1] / values[0]) ** (1.0 / month_num) - 1
    sql = "update portfolios set pf_month_returns = '%f' where p_risk = '%f' and pf_date = '%s'" % (month_returns, 1.0 * key / 10, '2015-09-11')
    cur.execute(sql)
cur.close()
conn.commit()


length = len(money_values)
month_num = 1.0 * length / 20    
month_returns = (money_values[length-1] / money_values[0]) ** (1.0 / month_num) - 1
sql = "update portfolios set pf_month_returns = '%f' where p_risk = '%f' and pf_date = '%s'" % (month_returns, 1.0 * 0.0 / 10, '2015-09-11')
cur = conn.cursor()
cur.execute(sql)
cur.close()
conn.commit()


conn.close()
